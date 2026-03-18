# code/data/make_dataset.py
"""
Create an interim cleaned dataset from a raw NYC 311 snapshot.

Input:
  data/raw/311_erm2-nwe9_YYYY-MM-DD.csv  (created by fetch_311.py)

Output:
  data/interim/311_cleaned_<snapshot_date>.parquet
  data/interim/311_cleaned_<snapshot_date>.meta.json

Cleaning performed:
- Normalize empty strings -> NaN
- Parse created_date and closed_date as datetimes
- Keep only rows with created_date AND closed_date present
- Enforce closed_date >= created_date
- Drop duplicate unique_key (keep last occurrence)
- Compute resolution_time_hours
- Drop extreme outliers by percentile OR hard cap (optional)

Run:
  python -m code.data.make_dataset
Optional args:
  python -m code.data.make_dataset --raw-file data/raw/311_erm2-nwe9_2026-03-13.csv
  python -m code.data.make_dataset --outlier-mode percentile --outlier-p 0.999
"""

from __future__ import annotations      # for dataclass features

import argparse     # for command-line argument parsing
import json     # for saving metadata as JSON
from dataclasses import dataclass, asdict   # for structured metadata reporting
from datetime import datetime       # for timestamps and date handling
from pathlib import Path        # for convenient file path handling
from typing import Optional     # for optional type hints

import numpy as np      
import pandas as pd

from code.config import RAW_DIR, INTERIM_DIR

# Define a dataclass for the cleaning report metadata
# This structured report will capture details about the cleaning process, 
# including row counts at each step, 
# outlier filtering parameters, 
# and the final output file.
@dataclass
class CleaningReport:
    raw_file: str
    created_at_utc: str
    rows_raw: int
    rows_after_empty_to_nan: int
    rows_after_datetime_parse: int
    rows_after_required_dates: int
    rows_after_time_sanity: int
    rows_after_dedup_unique_key: int
    rows_after_target_compute: int
    outlier_mode: str
    outlier_param: float
    rows_after_outlier_filter: int
    columns_kept: list[str]
    output_file: str

# get the latest snapshot CSV in the raw directory, based on naming pattern
def _latest_snapshot_csv(raw_dir: Path) -> Path:
    # expects fetch naming like: 311_erm2-nwe9_YYYY-MM-DD.csv
    candidates = sorted(raw_dir.glob("311_erm2-nwe9_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No snapshot CSV found in {raw_dir}. Run fetch_311.py first.")
    return candidates[-1]

# extract date tag from filename for naming the interim output, e.g. 2026-03-13 from 311_erm2-nwe9_2026-03-13.csv
def _extract_date_tag(path: Path) -> str:
    # pulls YYYY-MM-DD from filename if present; otherwise uses today's date
    name = path.stem  # no suffix
    parts = name.split("_")
    if parts and len(parts[-1]) == 10 and parts[-1][4] == "-" and parts[-1][7] == "-":
        return parts[-1]
    return datetime.utcnow().date().isoformat()

# convert empty or whitespace-only strings to NaN for all object columns
def _empty_strings_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    # Convert whitespace-only strings to NaN.
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) == 0:
        return df
    df[obj_cols] = df[obj_cols].apply(lambda s: s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan}))
    return df

# parse specified columns as datetimes, coercing errors to NaT
def _parse_datetimes(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df

# Drop duplicate unique_key, keeping the last occurrence.
# If unique_key is not present, do nothing.
def _dedup_unique_key(df: pd.DataFrame) -> pd.DataFrame:
    if "unique_key" not in df.columns:
        return df
    # keep the last one if there is a duplicate
    return df.sort_values("unique_key").drop_duplicates(subset=["unique_key"], keep="last")

# Compute resolution_time_hours = (closed_date - created_date) in hours
def _compute_resolution_hours(df: pd.DataFrame) -> pd.DataFrame:
    # resolution_time_hours = (closed - created) in hours
    df["resolution_time_hours"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600.0
    return df

# Filter outliers in resolution_time_hours based on the specified mode and parameter.
def _filter_outliers(
    df: pd.DataFrame,
    mode: str,
    param: float,
) -> pd.DataFrame:
    """
    mode:
      - "none"
      - "percentile": drop rows with resolution_time_hours > quantile(param), param in (0,1)
      - "cap_hours": drop rows with resolution_time_hours > param (hours)
    """
    if mode == "none":
        return df

    y = df["resolution_time_hours"]
    if mode == "percentile":
        if not (0.0 < param < 1.0):
            raise ValueError("--outlier-p must be between 0 and 1 (e.g., 0.999).")
        cutoff = float(y.quantile(param))
        return df[y <= cutoff].copy()

    if mode == "cap_hours":
        cutoff = float(param)
        return df[y <= cutoff].copy()

    raise ValueError(f"Unknown outlier mode: {mode}")

# Main function to orchestrate the dataset creation process
def main():
    # Set up command-line argument parsing to allow customization of the cleaning operation,
    #  such as specifying a particular raw snapshot file,
    #  choosing which columns to keep, and configuring outlier filtering.
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-file", type=str, default="", help="Path to a raw snapshot CSV. If empty, uses latest.")
    parser.add_argument(
        "--keep-columns",
        type=str,
        default="",
        help=(
            "Comma-separated column subset to keep (optional). "
            "If empty, keeps all columns + computed target."
        ),
    )
    parser.add_argument(
        "--outlier-mode",
        type=str,
        default="percentile",
        choices=["none", "percentile", "cap_hours"],
        help="Outlier filtering mode for resolution_time_hours.",
    )
    parser.add_argument(
        "--outlier-p",
        type=float,
        default=0.999,
        help="If outlier-mode=percentile, quantile threshold (e.g., 0.999).",
    )
    parser.add_argument(
        "--outlier-cap-hours",
        type=float,
        default=24.0 * 365.0,  # 1 year
        help="If outlier-mode=cap_hours, maximum allowed hours.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Interim output format.",
    )
    args = parser.parse_args()

    # Determine the raw snapshot CSV to use, either from the provided argument or by finding the latest one in the RAW_DIR.
    raw_path = Path(args.raw_file) if args.raw_file else _latest_snapshot_csv(RAW_DIR)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(raw_path, low_memory=False)
    rows_raw = len(df)

    # Normalize empties -> NaN
    df = _empty_strings_to_nan(df)
    rows_after_empty = len(df)

    # Parse datetimes needed for target
    df = _parse_datetimes(df, ["created_date", "closed_date"])
    rows_after_dt = len(df)

    # Filter to rows where target can be computed
    if "created_date" not in df.columns or "closed_date" not in df.columns:
        raise KeyError("Expected columns created_date and closed_date not found in the raw snapshot.")

    # created_date and closed_date must be present to compute target
    df = df[df["created_date"].notna() & df["closed_date"].notna()].copy()
    rows_required_dates = len(df)

    # closed >= created
    df = df[df["closed_date"] >= df["created_date"]].copy()
    rows_sanity = len(df)

    # Dedup
    df = _dedup_unique_key(df)
    rows_dedup = len(df)

    # Compute target
    df = _compute_resolution_hours(df)
    # Remove any rows where target became NaN or negative 
    df = df[df["resolution_time_hours"].notna() & (df["resolution_time_hours"] >= 0)].copy()
    rows_after_target = len(df)

    # Optional: keep only a subset of columns
    keep_cols: Optional[list[str]] = None
    if args.keep_columns.strip():
        keep_cols = [c.strip() for c in args.keep_columns.split(",") if c.strip()]
        # Always keep required fields + target if present
        for required in ["unique_key", "created_date", "closed_date", "resolution_time_hours"]:
            if required in df.columns and required not in keep_cols:
                keep_cols.append(required)
        df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Outlier filtering
    if args.outlier_mode == "percentile":
        df = _filter_outliers(df, "percentile", args.outlier_p)
        outlier_param = args.outlier_p
    elif args.outlier_mode == "cap_hours":
        df = _filter_outliers(df, "cap_hours", args.outlier_cap_hours)
        outlier_param = args.outlier_cap_hours
    else:
        outlier_param = 0.0

    rows_after_outliers = len(df)

    # Save
    date_tag = _extract_date_tag(raw_path)
    base = INTERIM_DIR / f"311_cleaned_{date_tag}"

    # Save the cleaned dataset in the specified format (Parquet or CSV) for efficient storage and future use.
    if args.output_format == "parquet":
        out_file = base.with_suffix(".parquet")
        df.to_parquet(out_file, index=False)
    else:
        out_file = base.with_suffix(".csv")
        df.to_csv(out_file, index=False)

    # Write metadata report, saves to data/interim/311_cleaned_<snapshot_date>.meta.json
    report = CleaningReport(
        raw_file=str(raw_path),
        created_at_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        rows_raw=rows_raw,
        rows_after_empty_to_nan=rows_after_empty,
        rows_after_datetime_parse=rows_after_dt,
        rows_after_required_dates=rows_required_dates,
        rows_after_time_sanity=rows_sanity,
        rows_after_dedup_unique_key=rows_dedup,
        rows_after_target_compute=rows_after_target,
        outlier_mode=args.outlier_mode,
        outlier_param=float(outlier_param),
        rows_after_outlier_filter=rows_after_outliers,
        columns_kept=list(df.columns),
        output_file=str(out_file),
    )

    meta_file = base.with_suffix(".meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    # print to console the paths to the saved interim dataset and the cleaning report
    print(f"Interim dataset saved: {out_file}")
    print(f"Cleaning report saved: {meta_file}")
    print(f"Rows: {rows_raw} -> {rows_after_outliers}")

if __name__ == "__main__":
    main()