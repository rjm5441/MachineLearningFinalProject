# code/utils/inspect_features.py
"""
Inspect preprocessing + fold dimensions

Writes a TXT report with:
- dataset sizes
- per-fold X_train/X_test dimensions after preprocessing
- target stats
- number of transformed features and sample feature names

Run:
  python -m code.utils.inspect_features
or:
  python -m code.utils.inspect_features --tag processed_2026-03-13_cv5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from code.utils.preprocessing import make_preprocessor

PROCESSED = Path("data/processed")
REPORTS = Path("reports/tables")


def _latest_processed_tag(processed_dir: Path) -> str:
    specs = sorted(processed_dir.glob("feature_spec_processed_*.json"))
    if not specs:
        raise FileNotFoundError("No processed feature_spec files found. Run build_features.py first.")
    return specs[-1].stem.replace("feature_spec_", "")


def _load_processed(tag: str) -> Tuple[pd.DataFrame, np.ndarray, np.lib.npyio.NpzFile, Dict]:
    Xraw = pd.read_parquet(PROCESSED / f"Xraw_{tag}.parquet")
    y = np.load(PROCESSED / f"y_{tag}.npy")
    folds = np.load(PROCESSED / f"folds_{tag}.npz")
    spec = json.loads((PROCESSED / f"feature_spec_{tag}.json").read_text())
    return Xraw, y, folds, spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="", help="Processed build tag, e.g. processed_2026-03-13_cv5")
    args = parser.parse_args()

    tag = args.tag if args.tag else _latest_processed_tag(PROCESSED)
    REPORTS.mkdir(parents=True, exist_ok=True)

    Xraw, y, folds, spec = _load_processed(tag)

    cat_cols = spec["categorical_cols"]
    num_cols = spec["numeric_cols"] + spec["time_feature_cols"]
    n_splits = int(spec["n_splits"])

    lines = []
    lines.append(f"Preprocessing Inspection Report (NO-LEAKAGE) — {tag}")
    lines.append("=" * 72)
    lines.append(f"Generated (UTC): {datetime.utcnow().isoformat(timespec='seconds')}Z")
    lines.append("")
    lines.append("SOURCE FILES")
    lines.append(f"- Xraw:   data/processed/Xraw_{tag}.parquet")
    lines.append(f"- y:      data/processed/y_{tag}.npy")
    lines.append(f"- folds:  data/processed/folds_{tag}.npz")
    lines.append(f"- spec:   data/processed/feature_spec_{tag}.json")
    lines.append("")
    lines.append("RAW DATASET")
    lines.append(f"- Rows: {len(Xraw)}")
    lines.append(f"- Raw feature columns: {Xraw.shape[1]}")
    lines.append(f"- Categorical cols ({len(cat_cols)}): {', '.join(cat_cols)}")
    lines.append(f"- Numeric+time cols ({len(num_cols)}): {', '.join(num_cols)}")
    lines.append("")
    lines.append("TARGET (y)")
    lines.append(f"- shape: {y.shape}")
    lines.append(f"- transform: {spec.get('target_transform', 'unknown')}")
    lines.append(
        f"- min/mean/median/max: "
        f"{float(np.min(y)):.4f} / {float(np.mean(y)):.4f} / {float(np.median(y)):.4f} / {float(np.max(y)):.4f}"
    )
    lines.append(f"- std: {float(np.std(y)):.4f}")
    lines.append("")

    # Fit once on fold 0 training to capture output dimension and feature names
    tr0 = folds["train_idx_0"]
    te0 = folds["test_idx_0"]
    pre0 = make_preprocessor(cat_cols, num_cols)
    Xtr0 = pre0.fit_transform(Xraw.iloc[tr0])
    Xte0 = pre0.transform(Xraw.iloc[te0])

    try:
        feat_names = pre0.get_feature_names_out()
        feat_count = len(feat_names)
        sample_names = list(feat_names[:40])
    except Exception:
        feat_count = int(Xtr0.shape[1])
        sample_names = []

    lines.append("TRANSFORMED FEATURE SPACE (from Fold 0 train-fit)")
    lines.append(f"- X_train (fold0) shape: {Xtr0.shape}")
    lines.append(f"- X_test  (fold0) shape: {Xte0.shape}")
    lines.append(f"- total transformed features: {feat_count}")
    if sample_names:
        lines.append("- sample transformed feature names:")
        for n in sample_names:
            lines.append(f"  - {n}")
    lines.append("")

    lines.append("PER-FOLD SHAPES (preprocessor fit on train only)")
    for i in range(n_splits):
        tr = folds[f"train_idx_{i}"]
        te = folds[f"test_idx_{i}"]

        pre = make_preprocessor(cat_cols, num_cols)
        X_train = pre.fit_transform(Xraw.iloc[tr])
        X_test = pre.transform(Xraw.iloc[te])

        lines.append(f"- Fold {i}:")
        lines.append(f"    train rows: {len(tr)}, test rows: {len(te)}")
        lines.append(f"    X_train shape: {X_train.shape}")
        lines.append(f"    X_test  shape: {X_test.shape}")

    out_path = REPORTS / f"preprocessing_inspection_{tag}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()