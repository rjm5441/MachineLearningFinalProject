# code/features/build_features.py
"""
feature build:
- Produces raw feature table (not encoded/scaled) + y + reproducible CV folds.
- Preprocessing is defined (column lists + intended transformers) but not fit here.

Inputs:
  data/interim/311_cleaned_<date>.parquet (or .csv)

Outputs (data/processed/):
  - Xraw_<tag>.parquet            Raw features (categorical + numeric + time-derived)
  - y_<tag>.npy                   Target vector aligned to Xraw rows
  - folds_<tag>.npz               CV indices (train_idx_i/test_idx_i)
  - feature_spec_<tag>.json       specifications (columns + transform settings)
  - feature_report_<tag>.md       report explaining the build

Run:
  python -m code.features.build_features
"""

from __future__ import annotations  # for future dataclass features

import argparse # for command-line argument parsing
import json # for saving metadata and feature specifications as JSON
from dataclasses import asdict, dataclass   # for defining a dataclass to hold feature specifications
from datetime import datetime   # for timestamping the feature build and naming outputs
from pathlib import Path    # for convenient a handling of file paths 
from typing import List, Tuple  # for type hints of lists and tuples

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype
from sklearn.model_selection import StratifiedKFold     # for creating stratified CV folds based on the target variable distribution

from code.config import INTERIM_DIR, PROCESSED_DIR, RANDOM_SEED

# Define the target column and candidate feature columns (both categorical and numeric) based on the interim dataset.
DATETIME_COL = "created_date"
TARGET_COL = "resolution_time_hours"

# Candidate feature columns to consider from the interim dataset.
#  These will be filtered to only those that exist in the loaded interim dataset.
CATEGORICAL_CANDIDATES = [
    "agency",
    "complaint_type",
    "descriptor",
    "borough",
    "location_type",
    "incident_zip",
    "community_board",
    "council_district",
    "police_precinct",
    "city",
]

# Numeric features that are already numeric in the interim dataset and can be used directly.
NUMERIC_CANDIDATES = [
    "latitude",
    "longitude",
]

# Time-derived features that will be created from the created_date column, if it exists and is parseable as a datetime.
TIME_FEATURES = ["created_hour", "created_dayofweek", "created_month", "created_is_weekend"]

# class to hold the feature build specifications for reproducibility and downstream use
@dataclass(frozen=True)
class FeatureSpec:
    tag: str
    source_interim: str
    created_at_utc: str

    # target
    target_col: str
    target_transform: str  # "none" or "log1p"

    # feature columns (raw)
    categorical_cols: List[str]
    numeric_cols: List[str]
    time_feature_cols: List[str]

    # CV
    n_splits: int
    strat_bins: int
    random_seed: int

    # intended preprocessing (not fit here)
    categorical_imputer: str
    categorical_encoder: str
    numeric_imputer: str
    numeric_scaler: str

# get the latest interim dataset in the interim directory, based on naming pattern
def _latest_interim(interim_dir: Path) -> Path:
    cands = sorted(interim_dir.glob("311_cleaned_*.parquet"))
    if cands:
        return cands[-1]
    cands = sorted(interim_dir.glob("311_cleaned_*.csv"))
    if cands:
        return cands[-1]
    raise FileNotFoundError(f"No interim dataset found in {interim_dir}. Run make_dataset.py first.")

# extract date tag from filename for naming the processed output, e.g. 2026-03-13 from 311_cleaned_2026-03-13.parquet
def _extract_date_tag(path: Path) -> str:
    # expects: 311_cleaned_YYYY-MM-DD.(parquet|csv)
    stem = path.stem
    parts = stem.split("_")
    if parts and len(parts[-1]) == 10 and parts[-1][4] == "-" and parts[-1][7] == "-":
        return parts[-1]
    return datetime.utcnow().date().isoformat()

# Load interim dataset from either parquet or CSV, depending on file extension.
def _load_interim(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported interim file type: {path.suffix}")

# Add time-derived features from the created_date column, if it exists and is parseable.
def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if DATETIME_COL not in df.columns:
        return df

    if not (is_datetime64_any_dtype(df[DATETIME_COL]) or is_datetime64tz_dtype(df[DATETIME_COL])):
        df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce", utc=True)

    df = df[df[DATETIME_COL].notna()].copy()
    df["created_hour"] = df[DATETIME_COL].dt.hour.astype("int16")
    df["created_dayofweek"] = df[DATETIME_COL].dt.dayofweek.astype("int16")
    df["created_month"] = df[DATETIME_COL].dt.month.astype("int16")
    df["created_is_weekend"] = (df["created_dayofweek"] >= 5).astype("int8")
    return df

# Create target vector y based on the specified target column and transformation, returning y and the aligned index.
def _make_target(df: pd.DataFrame, transform: str) -> Tuple[np.ndarray, pd.Index]:
    # Ensure the target column exists in the dataset, and convert it to numeric, coercing errors to NaN and dropping infinite values.
    if TARGET_COL not in df.columns:
        raise KeyError(f"Expected `{TARGET_COL}` in interim dataset.")
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # get the index of non-missing target values to align with the feature table, and convert y to a numpy array of type float64 for modeling.
    idx = y.dropna().index
    y = y.loc[idx].astype(float)

    # Apply the specified transformation to y. If "none", return as is. If "log1p", apply log(1+y) after clipping to ensure non-negativity. 
    # Raise an error for unknown transforms.
    if transform == "none":
        return y.to_numpy(dtype="float64"), idx
    if transform == "log1p":
        y = y.clip(lower=0)
        return np.log1p(y.to_numpy(dtype="float64")), idx
    raise ValueError(f"Unknown target transform: {transform}")

# Create stratification bins for y using quantiles, with a fallback to fewer bins if there are too many duplicates.
# this is used for stratified CV splitting to ensure each fold has a similar distribution of the target variable.
def _make_strat_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    s = pd.Series(y)
    try:
        bins = pd.qcut(s, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.qcut(s, q=min(5, n_bins), labels=False, duplicates="drop")
    return bins.to_numpy()

# Write a markdown report summarizing the feature build specifications and CV fold sizes.
def _write_report_md(path: Path, spec: FeatureSpec, n_rows: int, fold_sizes: List[Tuple[int, int]]) -> None:
    lines = []
    lines.append(f"# Feature Build Report — {spec.tag}")
    lines.append("")
    lines.append("## Results:")
    lines.append("- Raw feature table (not encoded/scaled)")
    lines.append("- Target vector aligned to the raw table")
    lines.append("- Reproducible CV fold indices")
    lines.append("")
    lines.append("## Source:")
    lines.append(f"- Interim file: `{spec.source_interim}`")
    lines.append(f"- Generated at (UTC): `{spec.created_at_utc}`")
    lines.append("")
    lines.append("## Target:")
    lines.append(f"- Column: `{spec.target_col}`")
    lines.append(f"- Transform: `{spec.target_transform}`")
    lines.append("")
    lines.append("## Feature Columns Used (raw):")
    lines.append(f"- Categorical ({len(spec.categorical_cols)}): {', '.join(spec.categorical_cols) if spec.categorical_cols else '(none)'}")
    lines.append(f"- Numeric ({len(spec.numeric_cols)}): {', '.join(spec.numeric_cols) if spec.numeric_cols else '(none)'}")
    lines.append(f"- Time-derived from `{DATETIME_COL}` ({len(spec.time_feature_cols)}): {', '.join(spec.time_feature_cols) if spec.time_feature_cols else '(none)'}")
    lines.append("")
    lines.append("## Intended Encoding / Scaling (fit per-fold to avoid leakage)")
    lines.append(f"- Categorical: {spec.categorical_imputer} + {spec.categorical_encoder}")
    lines.append(f"- Numeric + time: {spec.numeric_imputer} + {spec.numeric_scaler}")
    lines.append("")
    lines.append("## CV Folds")
    lines.append(f"- CV: StratifiedKFold(n_splits={spec.n_splits}, shuffle=True, random_state={spec.random_seed})")
    lines.append(f"- Stratification: quantile bins of y (requested bins = {spec.strat_bins})")
    lines.append("- Fold sizes (train, test):")
    for i, (tr, te) in enumerate(fold_sizes):
        lines.append(f"  - Fold {i}: ({tr}, {te})")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Preprocessing must be fit on the training split inside each fold.")
    lines.append("- All models should use the same folds + same preprocessing definition for fair comparison.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    # Set up command-line argument parsing to allow customization of the feature build operation,
    #  such as specifying a particular interim file,
    #  configuring the target transformation, 
    # and setting a maximum number of rows for quick experiments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--interim-file", type=str, default="", help="Interim cleaned dataset path. If empty, uses latest.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--bins", type=int, default=10, help="Quantile bins for stratification.")
    parser.add_argument("--target-transform", type=str, default="log1p", choices=["none", "log1p"])
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap for quick experiments (0 = no cap).")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Determine the interim dataset to use, either from the provided argument or by finding the latest one in the INTERIM_DIR. Load it into a DataFrame.
    interim_path = Path(args.interim_file) if args.interim_file else _latest_interim(INTERIM_DIR)
    df = _load_interim(interim_path)

    # Add time-derived features from the created_date column, if it exists and is parseable. 
    # This adds to the feature set with potentially useful information.
    df = _add_time_features(df)

    # If a maximum number of rows is specified for quick experiments, 
    # sample that many rows from the dataset using a fixed random seed for reproducibility.
    if args.max_rows and args.max_rows > 0:
        df = df.sample(n=min(args.max_rows, len(df)), random_state=RANDOM_SEED).copy()

    # Create the target vector y based on the specified target column and transformation, 
    # and get the aligned index to ensure that y and the feature table are properly aligned for modeling.
    y, idx = _make_target(df, args.target_transform)
    df = df.loc[idx].copy()

    # Select columns that exist
    cat_cols = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    num_cols = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    time_cols = [c for c in TIME_FEATURES if c in df.columns]

    # Ensure numeric are numeric dtype
    for c in num_cols + time_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Leakage guard: never include closed_date or target in features
    for leak in ["closed_date", TARGET_COL]:
        if leak in cat_cols:
            cat_cols.remove(leak)
        if leak in num_cols:
            num_cols.remove(leak)
        if leak in time_cols:
            time_cols.remove(leak)

    # Raw X table
    Xraw = df[cat_cols + num_cols + time_cols].copy()

    # CV folds (stratified by binned y)
    y_bins = _make_strat_bins(y, args.bins)
    # Create stratified CV folds using StratifiedKFold, ensuring that each fold has a similar distribution of the target variable.
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_SEED)

    # Store the train and test indices for each fold in a dictionary for later saving, and also keep track of the sizes of each fold for reporting.
    fold_dict = {}
    # fold_sizes will hold tuples of (train_size, test_size) for each fold, which will be included in the feature build report for transparency about the data splits.
    fold_sizes: List[Tuple[int, int]] = []
    # Loop through the folds generated by StratifiedKFold, storing the train and test indices for each fold in the fold_dict with keys like "train_idx_0", "test_idx_0", etc., and also recording the sizes of the train and test splits for reporting purposes. 
    # The indices are converted to int64 for consistency when saving to .npz format later.
    for i, (tr, te) in enumerate(skf.split(np.zeros_like(y), y_bins)):
        fold_dict[f"train_idx_{i}"] = tr.astype(np.int64)
        fold_dict[f"test_idx_{i}"] = te.astype(np.int64)
        fold_sizes.append((len(tr), len(te)))

    date_tag = _extract_date_tag(interim_path)
    tag = f"processed_{date_tag}_cv{args.n_splits}"

    # Save artifacts
    x_path = PROCESSED_DIR / f"Xraw_{tag}.parquet"
    y_path = PROCESSED_DIR / f"y_{tag}.npy"
    folds_path = PROCESSED_DIR / f"folds_{tag}.npz"
    spec_path = PROCESSED_DIR / f"feature_spec_{tag}.json"
    report_path = PROCESSED_DIR / f"feature_report_{tag}.md"
    
    # Save the raw feature table as a Parquet file for efficient storage, 
    # the target vector as a NumPy array, 
    # and the CV fold indices in a compressed .npz format.
    Xraw.to_parquet(x_path, index=False)
    np.save(y_path, y)
    np.savez_compressed(folds_path, **fold_dict)

    # Save the feature specification (columns + transform settings) as JSON 
    # for reproducibility and downstream use.
    spec = FeatureSpec(
        tag=tag,
        source_interim=str(interim_path),
        created_at_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        target_col=TARGET_COL,
        target_transform=args.target_transform,
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
        time_feature_cols=time_cols,
        n_splits=args.n_splits,
        strat_bins=args.bins,
        random_seed=RANDOM_SEED,
        categorical_imputer="SimpleImputer(most_frequent)",
        categorical_encoder="OneHotEncoder(handle_unknown='ignore')",
        numeric_imputer="SimpleImputer(median)",
        numeric_scaler="StandardScaler",
    )

    # Save the feature specification as a JSON file for record-keeping.
    spec_path.write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")
    _write_report_md(report_path, spec, n_rows=len(Xraw), fold_sizes=fold_sizes)

    # print to console results and where they were saved, along with the number of rows and raw columns in the feature table
    print("Saved:")
    print(f"- Xraw:   {x_path}")
    print(f"- y:      {y_path}")
    print(f"- folds:  {folds_path}")
    print(f"- spec:   {spec_path}")
    print(f"- report: {report_path}")
    print(f"Rows: {len(Xraw)}; Raw columns: {Xraw.shape[1]}")


if __name__ == "__main__":
    main()