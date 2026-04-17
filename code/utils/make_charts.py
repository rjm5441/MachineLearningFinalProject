# code/utils/make_charts.py
"""
Generate charts for the project presentation.

Inputs:
- reports/preds/preds_random_forest_regressor_BEST_processed_2026-03-13_cv5_head.csv
- reports/tables/rf_results_processed_2026-03-13_cv5.csv
- data/processed/Xraw_processed_2026-03-13_cv5.parquet
- data/processed/y_processed_2026-03-13_cv5.npy

Outputs:
- PNG figures saved into: reports/figures/

"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# PATHS
PRED_PARQUET = Path("reports/preds/preds_random_forest_regressor_BEST_processed_2026-03-13_cv5.parquet")
RF_RESULTS_CSV = Path("reports/tables/rf_results_processed_2026-03-13_cv5.csv")
XRAW_PARQUET = Path("data/processed/Xraw_processed_2026-03-13_cv5.parquet")
Y_NPY = Path("data/processed/y_processed_2026-03-13_cv5.npy")

OUT_DIR = Path("reports/figures")

# --- Slide theme colors ---
THEME = {
    "bg": "#2A2A2A",
    "yellow": "#F9DA78",
    "coral": "#EC736F",
    "white": "#FFFFFF",
    "gray_light": "#C4C2C1",
    "gray_mid": "#777267",
}

# Global matplotlib styling to match the deck
plt.rcParams.update({
    "figure.facecolor": THEME["bg"],
    "axes.facecolor": THEME["bg"],
    "savefig.facecolor": THEME["bg"],
    "text.color": THEME["white"],
    "axes.labelcolor": THEME["white"],
    "xtick.color": THEME["white"],
    "ytick.color": THEME["white"],
    "axes.edgecolor": THEME["gray_light"],
    "grid.color": THEME["gray_mid"],
    "axes.titleweight": "bold",
})

# default line/bar cycle (keeps charts consistent)
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[
    THEME["yellow"], THEME["coral"], THEME["gray_light"], THEME["white"]
])

def _ensure_outdir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(name: str) -> None:
    out = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=300, facecolor=THEME["bg"])
    plt.close()
    print(f"Saved: {out}")


def _make_model_charts(
    model_label: str,
    prefix: str,
    preds_path: Path,
    results_path: Path,
) -> None:
    """
    Generate the standard suite of charts for any trained model.

    - model_label : display name used in plot titles, e.g. "MLP"
    - prefix      : short string used in output filenames, e.g. "mlp"
    - preds_path  : parquet with columns y_true / y_pred (all folds stacked)
    - results_path: CSV with fold rows + summary rows (row_type column)
    """
    if not preds_path.exists():
        print(f"Skipping {model_label} charts: {preds_path} not found.")
        return
    if not results_path.exists():
        print(f"Skipping {model_label} charts: {results_path} not found.")
        return

    preds = pd.read_parquet(preds_path)
    results = pd.read_csv(results_path)

    # --- 1) Pred vs true scatter ---
    lo = float(min(preds["y_true"].min(), preds["y_pred"].min()))
    hi = float(max(preds["y_true"].max(), preds["y_pred"].max()))
    plt.figure()
    plt.scatter(preds["y_true"], preds["y_pred"], s=8, alpha=0.6, color=THEME["yellow"])
    plt.plot([lo, hi], [lo, hi], color=THEME["coral"], linewidth=1.5, label="perfect")
    plt.title(f"Best {model_label}: Predictions vs Actual (log1p scale)")
    plt.xlabel("y_true (log1p(hours))")
    plt.ylabel("y_pred (log1p(hours))")
    plt.legend()
    _savefig(f"{prefix}_pred_vs_true_scatter.png")

    # --- 2) Residual histogram ---
    resid = preds["y_pred"] - preds["y_true"]
    plt.figure()
    plt.hist(resid, bins=60, color=THEME["yellow"])
    plt.axvline(0, color=THEME["coral"], linewidth=1.5, linestyle="--")
    plt.title(f"Best {model_label}: Residuals (y_pred − y_true, log1p scale)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    _savefig(f"{prefix}_residual_hist.png")

    # --- 3) Absolute error vs target ---
    abs_err = resid.abs()
    plt.figure()
    plt.scatter(preds["y_true"], abs_err, s=8, alpha=0.5, color=THEME["coral"])
    plt.title(f"Best {model_label}: |Error| vs Target (log1p scale)")
    plt.xlabel("y_true (log1p(hours))")
    plt.ylabel("|y_pred − y_true|")
    _savefig(f"{prefix}_abs_error_vs_target.png")

    # --- 4) Fold-level RMSE and MAE for best config ---
    if "row_type" not in results.columns:
        print(f"  Warning: {results_path} missing 'row_type' column; skipping fold charts.")
        return

    summary = results[results["row_type"] == "summary"].copy()
    is_best_mask = summary["is_best"].astype(str).str.lower().isin(["true", "1"])
    if is_best_mask.any():
        best_cfg = summary.loc[is_best_mask, "config_id"].iloc[0]
    else:
        best_cfg = summary.sort_values("rmse_mean", ascending=True)["config_id"].iloc[0]

    fold_rows = results[(results["row_type"] == "fold") & (results["config_id"] == best_cfg)].copy()
    fold_rows = fold_rows.sort_values("fold")

    plt.figure()
    plt.plot(fold_rows["fold"], fold_rows["rmse"], marker="o", color=THEME["yellow"])
    plt.title(f"{model_label} Fold RMSE — Best Config: {best_cfg}")
    plt.xlabel("Fold")
    plt.ylabel("RMSE (log1p scale)")
    plt.grid(True, alpha=0.3)
    _savefig(f"{prefix}_fold_rmse.png")

    plt.figure()
    plt.plot(fold_rows["fold"], fold_rows["mae"], marker="o", color=THEME["coral"])
    plt.title(f"{model_label} Fold MAE — Best Config: {best_cfg}")
    plt.xlabel("Fold")
    plt.ylabel("MAE (log1p scale)")
    plt.grid(True, alpha=0.3)
    _savefig(f"{prefix}_fold_mae.png")

    # --- 5) Hyperparameter search: mean RMSE by config ---
    summary_sorted = summary.sort_values("rmse_mean", ascending=True).copy()
    labels = summary_sorted["config_id"].astype(str).tolist()
    rmse_vals = summary_sorted["rmse_mean"].to_numpy()
    colors = [THEME["coral"] if str(c) == str(best_cfg) else THEME["yellow"] for c in labels]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), rmse_vals, color=colors)
    plt.title(f"{model_label} Hyperparameter Search: Mean RMSE by Config (lower is better)")
    plt.ylabel("Mean RMSE (log1p scale)")
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    _savefig(f"{prefix}_mean_rmse_by_config.png")


def main():
    _ensure_outdir()

    
    # Load data
    if not PRED_PARQUET.exists():
        raise FileNotFoundError(f"Missing predictions parquet: {PRED_PARQUET}")
    if not RF_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing RF results CSV: {RF_RESULTS_CSV}")
    if not XRAW_PARQUET.exists():
        raise FileNotFoundError(f"Missing Xraw parquet: {XRAW_PARQUET}")
    if not Y_NPY.exists():
        raise FileNotFoundError(f"Missing y npy: {Y_NPY}")

    preds = pd.read_parquet(PRED_PARQUET)
    rf = pd.read_csv(RF_RESULTS_CSV)
    Xraw = pd.read_parquet(XRAW_PARQUET)
    y = np.load(Y_NPY)

    # 1) Target distribution (y is log1p(hours))

    plt.figure()
    plt.hist(y, bins=60)
    plt.title("Target Distribution: y = log1p(resolution_time_hours)")
    plt.xlabel("y (log1p(hours))")
    plt.ylabel("Count")
    _savefig("target_hist_log1p.png")

    # Back-transformed distribution (hours), clipped for readability
    y_hours = np.expm1(y)
    y_hours_clip = np.clip(y_hours, 0, np.quantile(y_hours, 0.99))
    plt.figure()
    plt.hist(y_hours_clip, bins=60)
    plt.title("Target Distribution: y = resolution_time_hours")
    plt.xlabel("resolution_time_hours (clipped)")
    plt.ylabel("Count")
    _savefig("target_hist_hours_clipped.png")

   
    # 2) Predictions quality: y_true vs y_pred + residuals
    needed_cols = {"y_true", "y_pred"}
    if not needed_cols.issubset(preds.columns):
        raise ValueError(f"Predictions CSV missing required columns: {needed_cols - set(preds.columns)}")

    # y_true vs y_pred scatter
    plt.figure()
    plt.scatter(preds["y_true"], preds["y_pred"], s=8, alpha=0.6)
    # diagonal reference
    lo = float(min(preds["y_true"].min(), preds["y_pred"].min()))
    hi = float(max(preds["y_true"].max(), preds["y_pred"].max()))
    plt.plot([lo, hi], [lo, hi])
    plt.title("Best RF: Predictions vs Actual (log1p scale)")
    plt.xlabel("y_true (log1p(hours))")
    plt.ylabel("y_pred (log1p(hours))")
    _savefig("pred_vs_true_scatter.png")

    # residual histogram
    resid = preds["y_pred"] - preds["y_true"]
    plt.figure()
    plt.hist(resid, bins=60)
    plt.title("Best RF: Residuals (y_pred - y_true, log1p scale)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    _savefig("residual_hist.png")

    # absolute error vs y_true 
    abs_err = (preds["y_pred"] - preds["y_true"]).abs()
    plt.figure()
    plt.scatter(preds["y_true"], abs_err, s=8, alpha=0.6)
    plt.title("Best RF: Absolute Error vs Target (log1p scale)")
    plt.xlabel("y_true (log1p(hours))")
    plt.ylabel("|y_pred - y_true|")
    _savefig("abs_error_vs_target.png")


    
    # 3) Fold stability (RMSE by fold) for best config
    # rf_results file includes both fold rows and summary rows.
    if "row_type" not in rf.columns:
        raise ValueError("rf_results CSV missing 'row_type' column (expected 'fold' and 'summary').")

    # Identify best config from summary rows (is_best == True)
    summary = rf[rf["row_type"] == "summary"].copy()
    if "is_best" in summary.columns and summary["is_best"].astype(str).str.lower().isin(["true", "1"]).any():
        best_cfg = summary.loc[summary["is_best"].astype(str).str.lower().isin(["true", "1"]), "config_id"].iloc[0]
    else:
        # fallback: best by rmse_mean
        best_cfg = summary.sort_values("rmse_mean", ascending=True)["config_id"].iloc[0]

    fold_rows = rf[(rf["row_type"] == "fold") & (rf["config_id"] == best_cfg)].copy()
    fold_rows = fold_rows.sort_values("fold")

    plt.figure()
    plt.plot(fold_rows["fold"], fold_rows["rmse"], marker="o")
    plt.title(f"RF Fold RMSE (Best Config: {best_cfg})")
    plt.xlabel("Fold")
    plt.ylabel("RMSE (log1p scale)")
    plt.grid(True, alpha=0.3)
    _savefig("best_rf_rmse_by_fold.png")

    plt.figure()
    plt.plot(fold_rows["fold"], fold_rows["mae"], marker="o")
    plt.title(f"RF Fold MAE (Best Config: {best_cfg})")
    plt.xlabel("Fold")
    plt.ylabel("MAE (log1p scale)")
    plt.grid(True, alpha=0.3)
    _savefig("best_rf_mae_by_fold.png")


    # 4) Hyperparameter comparison chart (mean RMSE by config)
    # Use summary rows sorted by rmse_mean
    summary_sorted = summary.sort_values("rmse_mean", ascending=True).copy()

    # Make labels compact
    labels = summary_sorted["config_id"].astype(str).tolist()
    rmse_mean = summary_sorted["rmse_mean"].to_numpy()

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), rmse_mean)
    plt.title("Random Forest Hyperparameter Search: Mean RMSE by Config (lower is better)")
    plt.ylabel("Mean RMSE (log1p scale)")
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    _savefig("rf_mean_rmse_by_config.png")

    # 5) Dataset EDA charts from Xraw
    # Top complaint types
    if "complaint_type" in Xraw.columns:
        top_ct = Xraw["complaint_type"].astype(str).value_counts().head(12)
        plt.figure(figsize=(10, 5))
        plt.bar(top_ct.index.astype(str), top_ct.values)
        plt.title("Top Complaint Types (count)")
        plt.ylabel("Count")
        plt.xticks(rotation=60, ha="right")
        _savefig("top_complaint_types.png")

    # Borough counts
    if "borough" in Xraw.columns:
        top_b = Xraw["borough"].astype(str).value_counts()
        plt.figure(figsize=(8, 4))
        plt.bar(top_b.index.astype(str), top_b.values)
        plt.title("Borough Distribution (count)")
        plt.ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        _savefig("borough_counts.png")

    # Avg target by borough (on log scale y)
    if "borough" in Xraw.columns and len(Xraw) == len(y):
        tmp = pd.DataFrame({"borough": Xraw["borough"].astype(str), "y": y})
        borough_mean = tmp.groupby("borough")["y"].mean().sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        plt.bar(borough_mean.index.astype(str), borough_mean.values)
        plt.title("Mean Target by Borough (y = log1p(hours))")
        plt.ylabel("Mean y")
        plt.xticks(rotation=30, ha="right")
        _savefig("mean_target_by_borough_log1p.png")

    # ------------------------------------------------------------------ #
    # MLP charts
    # ------------------------------------------------------------------ #
    _make_model_charts(
        model_label="MLP",
        prefix="mlp",
        preds_path=Path("reports/preds/preds_mlp_regressor_BEST_processed_2026-03-13_cv5.parquet"),
        results_path=Path("reports/tables/mlp_results_processed_2026-03-13_cv5.csv"),
    )

    # ------------------------------------------------------------------ #
    # ResNet (Neural Net) charts
    # ------------------------------------------------------------------ #
    _make_model_charts(
        model_label="ResNet",
        prefix="resnet",
        preds_path=Path("reports/preds/preds_resnet_regressor_BEST_processed_2026-03-13_cv5.parquet"),
        results_path=Path("reports/tables/resnet_results_processed_2026-03-13_cv5.csv"),
    )

    print("\nDone. Figures written to reports/figures/.")


if __name__ == "__main__":
    main()