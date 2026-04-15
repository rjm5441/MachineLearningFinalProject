# code/models/train_neural_net.py
"""
DEEP RESIDUAL NEURAL NETWORK REGRESSION

Architecture: fully-connected residual network with batch normalization.
Each residual block applies: Linear -> BN -> ReLU -> Dropout -> Linear -> BN,
then adds the input (skip connection) before the final ReLU. This is deeper and
more expressive than the plain MLP in train_mlp.py and better suited to
high-dimensional tabular data with many one-hot encoded features.

- model training script:
    - Loads Xraw, y, folds, and feature_spec from data/processed/
    - Tries a small grid of hyperparameters (hidden_size, n_blocks, learning_rate, dropout)
    - For each config:
        - For each fold: fit preprocessing on train only, transform, train model, predict
        - Computes MAE and RMSE per fold
    - Selects the best config by lowest mean RMSE (ties broken by mean MAE)
    - Saves:
        1) one predictions file for the best config only (all folds stacked)
        2) one results CSV covering all tried configs (fold + summary rows), best marked clearly

Run:
  python -m code.models.train_neural_net --tag processed_2026-03-13_cv5
or:
  python -m code.models.train_neural_net
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def make_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    """Imputes + one-hot encodes categoricals; imputes + standardizes numerics. Fit on train only."""
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[("cat", cat_pipe, categorical_cols), ("num", num_pipe, numeric_cols)],
        remainder="drop",
        sparse_threshold=0.3,
    )


# Constants for file paths
PROCESSED = Path("data/processed")
REPORTS_TABLES = Path("reports/tables")
REPORTS_PREDS = Path("reports/preds")

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    """
    One residual block: Linear -> BN -> ReLU -> Dropout -> Linear -> BN,
    with a skip connection added before the final ReLU.
    If input and output dimensions differ, a 1x1 linear projection aligns them.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        # Project skip connection if dimensions differ
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + self.skip(x))


class ResNet(nn.Module):
    """
    Input projection -> stack of ResidualBlocks -> linear output head.
    """
    def __init__(self, input_size: int, hidden_size: int, n_blocks: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_size, hidden_size, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x)


@dataclass
class FoldResult:
    model: str
    tag: str
    config_id: str
    fold: int
    n_train: int
    n_test: int
    hidden_size: int
    n_blocks: int
    learning_rate: float
    dropout: float
    batch_size: int
    best_epoch: int
    mae: float
    rmse: float


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


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _to_dense(X) -> np.ndarray:
    if scipy.sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _build_grid() -> List[Dict]:
    """
    Hyperparameter grid for the residual network.
    - hidden_size: width of every hidden layer
    - n_blocks: number of residual blocks (depth)
    - learning_rate: Adam step size
    - dropout: applied inside each residual block
    - batch_size: fixed across all configs
    """
    hidden_sizes = [256]
    n_blocks_list = [2, 4]
    learning_rates = [1e-3, 3e-4]
    dropouts = [0.1, 0.3]
    batch_size = 512

    grid: List[Dict] = []
    for h in hidden_sizes:
        for nb in n_blocks_list:
            for lr in learning_rates:
                for d in dropouts:
                    grid.append({
                        "config_id": f"resnet_h{h}_b{nb}_lr{lr:.0e}_do{d}",
                        "hidden_size": h,
                        "n_blocks": nb,
                        "learning_rate": lr,
                        "dropout": d,
                        "batch_size": batch_size,
                    })
    return grid


def _train_resnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_size: int,
    n_blocks: int,
    learning_rate: float,
    dropout: float,
    batch_size: int,
    max_epochs: int = 100,
    patience: int = 10,
) -> Tuple[ResNet, int]:
    """
    Train a ResNet with early stopping on validation MSE.
    Returns the best model (restored to best-epoch weights) and that epoch number.
    """
    input_size = X_train.shape[1]
    model = ResNet(input_size=input_size, hidden_size=hidden_size, n_blocks=n_blocks, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Reduce LR by 0.5 if val loss plateaus for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    train_dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    best_val_loss = float("inf")
    best_state: Dict = {}
    best_epoch = 1
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch


def _predict(model: ResNet, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            batch = X_t[start: start + batch_size].to(DEVICE)
            preds.append(model(batch).cpu().squeeze(1).numpy())
    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="", help="Processed build tag, e.g. processed_2026-03-13_cv5")
    args = parser.parse_args()

    tag = args.tag if args.tag else _latest_processed_tag(PROCESSED)
    print(f"Using tag: {tag}")
    print(f"Device: {DEVICE}")

    Xraw, y, folds, spec = _load_processed(tag)

    cat_cols = spec["categorical_cols"]
    num_cols = spec["numeric_cols"] + spec["time_feature_cols"]
    n_splits = int(spec["n_splits"])

    REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS_PREDS.mkdir(parents=True, exist_ok=True)

    model_name = "resnet_regressor"

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1) GRID SEARCH
    grid = _build_grid()
    all_fold_results: List[FoldResult] = []

    for g in grid:
        config_id = g["config_id"]
        print(f"Processing config: {config_id} at time: {pd.Timestamp.now()}")

        for i in range(n_splits):
            tr = folds[f"train_idx_{i}"]
            te = folds[f"test_idx_{i}"]

            pre = make_preprocessor(cat_cols, num_cols)
            X_train_full = _to_dense(pre.fit_transform(Xraw.iloc[tr]))
            X_test = _to_dense(pre.transform(Xraw.iloc[te]))

            # 10% of training fold held out for early stopping
            tr_idx, val_idx = train_test_split(
                np.arange(len(X_train_full)), test_size=0.1, random_state=RANDOM_SEED
            )
            X_tr, y_tr = X_train_full[tr_idx], y[tr][tr_idx]
            X_val, y_val = X_train_full[val_idx], y[tr][val_idx]

            trained_model, best_epoch = _train_resnet(
                X_tr, y_tr, X_val, y_val,
                hidden_size=g["hidden_size"],
                n_blocks=g["n_blocks"],
                learning_rate=g["learning_rate"],
                dropout=g["dropout"],
                batch_size=g["batch_size"],
            )

            pred = _predict(trained_model, X_test)

            mae = float(mean_absolute_error(y[te], pred))
            rmse = _rmse(y[te], pred)

            all_fold_results.append(
                FoldResult(
                    model=model_name,
                    tag=tag,
                    config_id=config_id,
                    fold=i,
                    n_train=len(tr),
                    n_test=len(te),
                    hidden_size=g["hidden_size"],
                    n_blocks=g["n_blocks"],
                    learning_rate=g["learning_rate"],
                    dropout=g["dropout"],
                    batch_size=g["batch_size"],
                    best_epoch=best_epoch,
                    mae=mae,
                    rmse=rmse,
                )
            )

    df_folds = pd.DataFrame([asdict(r) for r in all_fold_results])

    # 2) SUMMARY + BEST SELECTION
    summary = (
        df_folds.groupby(
            ["model", "tag", "config_id", "hidden_size", "n_blocks", "learning_rate", "dropout", "batch_size"],
            dropna=False,
        )
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
        )
        .reset_index()
    )

    summary_sorted = summary.sort_values(["rmse_mean", "mae_mean"], ascending=[True, True]).reset_index(drop=True)
    best_row = summary_sorted.iloc[0]
    best_config_id = best_row["config_id"]
    summary_sorted["is_best"] = summary_sorted["config_id"].eq(best_config_id)

    # 3) SAVE RESULTS TO CSV
    out_csv = REPORTS_TABLES / f"resnet_results_{tag}.csv"

    df_folds_out = df_folds.copy()
    df_folds_out["row_type"] = "fold"
    df_folds_out["is_best"] = df_folds_out["config_id"].eq(best_config_id)

    summary_out = summary_sorted.copy()
    summary_out["row_type"] = "summary"

    df_out = pd.concat([df_folds_out, summary_out], ignore_index=True, sort=False)
    df_out.to_csv(out_csv, index=False)

    # 4) SAVE PREDICTIONS FOR BEST MODEL ONLY
    best_preds_rows = []

    best_params = {
        "hidden_size": int(best_row["hidden_size"]),
        "n_blocks": int(best_row["n_blocks"]),
        "learning_rate": float(best_row["learning_rate"]),
        "dropout": float(best_row["dropout"]),
        "batch_size": int(best_row["batch_size"]),
    }

    for i in range(n_splits):
        tr = folds[f"train_idx_{i}"]
        te = folds[f"test_idx_{i}"]

        pre = make_preprocessor(cat_cols, num_cols)
        X_train_full = _to_dense(pre.fit_transform(Xraw.iloc[tr]))
        X_test = _to_dense(pre.transform(Xraw.iloc[te]))

        tr_idx, val_idx = train_test_split(
            np.arange(len(X_train_full)), test_size=0.1, random_state=RANDOM_SEED
        )
        X_tr, y_tr = X_train_full[tr_idx], y[tr][tr_idx]
        X_val, y_val = X_train_full[val_idx], y[tr][val_idx]

        best_model, _ = _train_resnet(
            X_tr, y_tr, X_val, y_val,
            hidden_size=best_params["hidden_size"],
            n_blocks=best_params["n_blocks"],
            learning_rate=best_params["learning_rate"],
            dropout=best_params["dropout"],
            batch_size=best_params["batch_size"],
        )
        pred = _predict(best_model, X_test)

        best_preds_rows.append(
            pd.DataFrame({
                "tag": tag,
                "model": model_name,
                "best_config_id": best_config_id,
                "fold": i,
                "y_true": y[te],
                "y_pred": pred,
            })
        )

    df_best_preds = pd.concat(best_preds_rows, ignore_index=True)
    out_pred = REPORTS_PREDS / f"preds_{model_name}_BEST_{tag}.parquet"
    df_best_preds.to_parquet(out_pred, index=False)

    # 5) PRINT QUICK SUMMARY
    print(f"\nGrid search complete for tag: {tag}")
    print(f"Results written to: {out_csv}")
    print(f"Best predictions written to: {out_pred}")
    print("\nBest configuration:")
    print(
        best_row[["config_id", "hidden_size", "n_blocks", "learning_rate", "dropout", "mae_mean", "rmse_mean"]].to_string(
            index=False
        )
    )
    print("\nAll configs by RMSE:")
    print(summary_sorted[["is_best", "config_id", "mae_mean", "rmse_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
