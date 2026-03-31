# code/models/train_tree.py
"""
RANDOM FOREST REGRESSION

- model training script:
    - Loads Xraw, y, folds, and feature_spec from data/processed/
    - Tries a small, sensible grid of RF hyperparameters (hard-coded)
    - For each config:
        - For each fold: fit preprocessing on train only, transform, train model, predict
        - Computes MAE and RMSE per fold
    - Selects the best config by lowest mean RMSE (ties broken by mean MAE)
    - Saves:
        1) one predictions file for the best config only (all folds stacked)
        2) one results CSV covering all tried configs (fold + summary rows), best marked clearly

Run:
  python -m code.models.train_tree --tag processed_2026-03-13_cv5
or:
  python -m code.models.train_tree
"""

from __future__ import annotations  # for data classes and type hints

import argparse  # for parsing command-line arguments
import json      # for loading feature_spec from JSON files
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from code.utils.preprocessing import make_preprocessor  # preprocessing pipeline 

# Constants for file paths
PROCESSED = Path("data/processed")
REPORTS_TABLES = Path("reports/tables")
REPORTS_PREDS = Path("reports/preds")


# Dataclass to store results for each fold/config
@dataclass
class FoldResult:
    model: str
    tag: str
    config_id: str
    fold: int
    n_train: int
    n_test: int
    n_estimators: int
    max_depth: Optional[int]
    min_samples_leaf: int
    mae: float
    rmse: float


# Helper functions to load the latest processed data based on the tag
def _latest_processed_tag(processed_dir: Path) -> str:
    # looks for feature_spec_processed_*.json and picks the latest
    specs = sorted(processed_dir.glob("feature_spec_processed_*.json"))
    if not specs:
        raise FileNotFoundError("No processed feature_spec files found. Run build_features.py first.")
    return specs[-1].stem.replace("feature_spec_", "")


# Returns Xraw, y, folds, spec for the given tag.
def _load_processed(tag: str) -> Tuple[pd.DataFrame, np.ndarray, np.lib.npyio.NpzFile, Dict]:
    Xraw = pd.read_parquet(PROCESSED / f"Xraw_{tag}.parquet")
    y = np.load(PROCESSED / f"y_{tag}.npy")
    folds = np.load(PROCESSED / f"folds_{tag}.npz")
    spec = json.loads((PROCESSED / f"feature_spec_{tag}.json").read_text())
    return Xraw, y, folds, spec

# RMSE helper
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# Builds a small grid of hyperparameters to try for the random forest.
def _build_grid() -> List[Dict]:
    """
    Notes:
    - n_estimators: num trees in the forest
    - max_depth: how deep each tree can grow; None means nodes are expanded until all leaves are pure or contain < min_samples_leaf samples
    - min_samples_leaf: the minimum number of samples required to be at a leaf node
    """
    n_estimators_list = [200, 300]
    max_depth_list = [20, 40]
    min_samples_leaf_list = [1, 5, 10]

    grid: List[Dict] = []
    for n in n_estimators_list:
        for d in max_depth_list:
            for leaf in min_samples_leaf_list:
                grid.append(
                    {
                        "config_id": f"rf_n{n}_d{d if d is not None else 'None'}_leaf{leaf}",
                        "n_estimators": n,
                        "max_depth": d,
                        "min_samples_leaf": leaf,
                    }
                )
    return grid


def main():
    # Command line argument for tag 
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="", help="Processed build tag, e.g. processed_2026-03-16_cv5")
    args = parser.parse_args()

    # check if tag is provided, otherwise find the latest processed tag in data/processed
    tag = args.tag if args.tag else _latest_processed_tag(PROCESSED)

    # Load the processed data (Xraw, y, folds, and feature_spec) for the specified tag.
    Xraw, y, folds, spec = _load_processed(tag)

    # Extract the categorical columns, numeric columns, time feature columns, and n_splits
    cat_cols = spec["categorical_cols"]
    num_cols = spec["numeric_cols"] + spec["time_feature_cols"]
    n_splits = int(spec["n_splits"])

    # Create directories for saving reports and predictions if they don't already exist.
    REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS_PREDS.mkdir(parents=True, exist_ok=True)

    model_name = "random_forest_regressor"

    # 1) GRID SEARCH
    grid = _build_grid()
    all_fold_results: List[FoldResult] = []

    # Iterate through each hyperparameter configuration in the grid 
    for g in grid:
        # Each config gets a unique config_id that encodes the hyperparameters, e.g. "rf_n100_dNone_leaf1"
        config_id = g["config_id"]
        # print the current time and config being processed for better tracking of long runs
        print(f"Processing config: {config_id} at time: {pd.Timestamp.now()}")
        # Iterate through each fold
        for i in range(n_splits):
            # Extract the train and test indices for the current fold from the folds object
            tr = folds[f"train_idx_{i}"]
            te = folds[f"test_idx_{i}"]

            # Build the preprocessing pipeline for this fold (fit on train only), then transform both train and test sets
            pre = make_preprocessor(cat_cols, num_cols)
            X_train = pre.fit_transform(Xraw.iloc[tr])
            X_test = pre.transform(Xraw.iloc[te])

            # build the model based on the hyperparam config
            model = RandomForestRegressor(
                n_estimators=g["n_estimators"],
                random_state=42,
                n_jobs=-1,
                max_depth=g["max_depth"],
                min_samples_leaf=g["min_samples_leaf"],
            )

            # fir the model, run the predictions
            model.fit(X_train, y[tr])
            pred = model.predict(X_test)

            # calculate error mae and rmse
            mae = float(mean_absolute_error(y[te], pred))
            rmse = _rmse(y[te], pred)

            # generate results for this fold and config, and append to the list of all results
            all_fold_results.append(
                FoldResult(
                    model=model_name,
                    tag=tag,
                    config_id=config_id,
                    fold=i,
                    n_train=len(tr),
                    n_test=len(te),
                    n_estimators=g["n_estimators"],
                    max_depth=g["max_depth"],
                    min_samples_leaf=g["min_samples_leaf"],
                    mae=mae,
                    rmse=rmse,
                )
            )
    # Convert the list of FoldResult dataclass instances into a DataFrame for easier analysis and saving.
    df_folds = pd.DataFrame([asdict(r) for r in all_fold_results])

    # 2) SUMMARY + BEST SELECTION
    # group by config and compute mean and std of mae and rmse across folds for each config
    summary = (
        df_folds.groupby(["model", "tag", "config_id", "n_estimators", "max_depth", "min_samples_leaf"], dropna=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
        )
        .reset_index()
    )

    # Choose best by RMSE mean, tie-break by MAE mean
    summary_sorted = summary.sort_values(["rmse_mean", "mae_mean"], ascending=[True, True]).reset_index(drop=True)
    best_row = summary_sorted.iloc[0]
    best_config_id = best_row["config_id"]

    # Add a clear marker column
    summary_sorted["is_best"] = summary_sorted["config_id"].eq(best_config_id)

    # 3) SAVE RESULTS TO CSV
   
    #  write a single CSV that includes:
    # - all fold-level rows
    # - plus a summary section at the end
    out_csv = REPORTS_TABLES / f"rf_results_{tag}.csv"

    # Create a type column to make it easy to filter through report
    df_folds_out = df_folds.copy()
    df_folds_out["row_type"] = "fold"
    # at is best column to make it easier to find the best one
    df_folds_out["is_best"] = df_folds_out["config_id"].eq(best_config_id)

    summary_out = summary_sorted.copy()
    summary_out["row_type"] = "summary"

    # Align columns across both frames
    # (fold rows have mae/rmse; summary rows have mae_mean/rmse_mean)
    # We'll keep both sets; missing values will be blank.
    df_out = pd.concat([df_folds_out, summary_out], ignore_index=True, sort=False)
    df_out.to_csv(out_csv, index=False)


    # 4) SAVE PREDICTIONS FOR BEST MODEL ONLY
    # Save ONE file containing predictions for best config across ALL folds.
    best_preds_rows = []

    # Pull best hyperparams (so we train best config again to generate preds)
    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": None if pd.isna(best_row["max_depth"]) else best_row["max_depth"],
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
    }
    # When max_depth is None, it can show up as NaN in a dataframe.
    if isinstance(best_row["max_depth"], float) and np.isnan(best_row["max_depth"]):
        best_params["max_depth"] = None

    # rerun the training loop for the best config to get predictions for all folds (stacked together in one file)
    for i in range(n_splits):
        tr = folds[f"train_idx_{i}"]
        te = folds[f"test_idx_{i}"]

        pre = make_preprocessor(cat_cols, num_cols)
        X_train = pre.fit_transform(Xraw.iloc[tr])
        X_test = pre.transform(Xraw.iloc[te])

        best_model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            random_state=42,
            n_jobs=-1,
            max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
        )
        best_model.fit(X_train, y[tr])
        pred = best_model.predict(X_test)

        best_preds_rows.append(
            pd.DataFrame(
                {
                    "tag": tag,
                    "model": model_name,
                    "best_config_id": best_config_id,
                    "fold": i,
                    "y_true": y[te],
                    "y_pred": pred,
                }
            )
        )

    df_best_preds = pd.concat(best_preds_rows, ignore_index=True)
    out_pred = REPORTS_PREDS / f"preds_{model_name}_BEST_{tag}.parquet"
    df_best_preds.to_parquet(out_pred, index=False)

    # 5) PRINT QUICK SUMMARY
   
    print(f"Grid search complete for tag: {tag}")
    print(f"Results written to: {out_csv}")
    print(f"Best predictions written to: {out_pred}")
    print("\nBest configuration:")
    print(best_row[["config_id", "n_estimators", "max_depth", "min_samples_leaf", "mae_mean", "rmse_mean"]].to_string(index=False))
    print("\nTop 5 configs by RMSE:")
    print(summary_sorted.head(5)[["is_best", "config_id", "mae_mean", "rmse_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
