import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import random

class SimpleTree:
    def __init__(self, max_depth, min_samples=50):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.feature_idx = None
        self.threshold = None
        self.left = None 
        self.right = None
        self.leaf_val = None  

    def _best_split(self, X, residuals):
        best_err, best_col, best_t = float('inf'), None, None

        for col in range(X.shape[1]):
            thresholds = np.unique(X[:, col])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, col], np.linspace(0, 100, 20))
            for t in thresholds:
                mask = X[:, col] <= t
                if mask.sum() < self.min_samples or (~mask).sum() < self.min_samples:
                    continue
                l_val = np.mean(residuals[mask])
                r_val = np.mean(residuals[~mask])
                err = np.sum((residuals[mask] - l_val)**2) + np.sum((residuals[~mask] - r_val)**2)
                if err < best_err:
                    best_err, best_col, best_t = err, col, t
        return best_col, best_t

    def fit(self, X, residuals, depth=0):
        # Stop if max depth reached or too few samples
        if depth >= self.max_depth or len(residuals) < self.min_samples * 2:
            self.leaf_val = np.mean(residuals)
            return

        col, t = self._best_split(X, residuals)
        if col is None:
            self.leaf_val = np.mean(residuals)
            return

        self.feature_idx, self.threshold = col, t
        mask = X[:, col] <= t

        self.left = SimpleTree(self.max_depth, self.min_samples)
        self.left.fit(X[mask], residuals[mask], depth + 1)

        self.right = SimpleTree(self.max_depth, self.min_samples)
        self.right.fit(X[~mask], residuals[~mask], depth + 1)

    def predict(self, X):
        if self.leaf_val is not None:
            return np.full(X.shape[0], self.leaf_val)
        mask = X[:, self.feature_idx] <= self.threshold
        preds = np.empty(X.shape[0])
        if np.any(mask):
            preds[mask] = self.left.predict(X[mask])
        if np.any(~mask):
            preds[~mask] = self.right.predict(X[~mask])
        return preds

class MyExtremeBooster:
    def __init__(self, n_estimators, lr, patience):

        self.n_estimators = n_estimators
        self.lr = lr
        self.patience = patience
        self.trees = []
        self.base_pred = 0

    def fit(self, X, y, eval_x=None, eval_y=None):
        self.base_pred = np.mean(y)
        current_preds = np.full(y.shape, self.base_pred)
        n_features = X.shape[1]

        best_val_mae = float('inf')
        no_improve_count = 0
        best_trees = []

        if eval_x is not None:
            val_preds = np.full(eval_y.shape, self.base_pred)
        
        for i in range(self.n_estimators):

            print(f"traning estimater {i} of {self.n_estimators}")
            residuals = y - current_preds
            
            # Randomly sample 60% of columns each round
            col_sample = np.random.choice(n_features, 
                                        max(1, int(n_features * 0.6)), 
                                        replace=False)
            X_subset = X[:, col_sample]

            tree = SimpleTree(max_depth=6)
            tree.fit(X_subset, residuals)
            
            current_preds += self.lr * tree.predict(X_subset)
            self.trees.append((tree, col_sample))  # store which cols were used

            # 1. Calculate Train Metric
            train_mae = np.mean(np.abs(y - current_preds))

            log_msg = f"Estimator {i} | Train MAE: {train_mae:.4f}"

            # 3. Add Val info ONLY if eval data exists
            if eval_x is not None:
                val_preds += self.lr * tree.predict(eval_x[:, col_sample])
                val_mae = np.mean(np.abs(eval_y - val_preds))
                log_msg += f" | Val MAE: {val_mae:.4f}"

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    no_improve_count = 0
                    best_trees = list(self.trees)  # snapshot best state
                else:
                    no_improve_count += 1
                    if no_improve_count >= self.patience:
                        print(log_msg)
                        print(f"Early stopping at estimator {i} "
                              f"(no improvement for {self.patience} rounds). "
                              f"Best val MAE: {best_val_mae:.4f}")
                        self.trees = best_trees  # roll back to best
                        break

            if i % 5 == 0:
                print(log_msg)

    def predict(self, X):
        y_hat = np.full(X.shape[0], self.base_pred)
        for tree, col_sample in self.trees:
            y_hat += self.lr * tree.predict(X[:, col_sample])
        return y_hat


def prepare_numpy_data(df):

    if 'created_hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['created_hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['created_hour'] / 24.0)
        df = df.drop(columns=['created_hour'])

    if 'created_dayofweek' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['created_dayofweek'] / 7.0)
        df['day_cos'] = np.cos(2 * np.pi * df['created_dayofweek'] / 7.0)
        df = df.drop(columns=['created_dayofweek'])

    # 1. One-Hot Encode 'agency'
    if 'agency' in df.columns:
        df = pd.get_dummies(df, columns=['agency'], prefix='agency', dtype=int)

    # 1. One-Hot Encode 'borough'
    if 'borough' in df.columns:
        df = pd.get_dummies(df, columns=['borough'], prefix='borough', dtype=int)

    if 'created_month' in df.columns:
        df = pd.get_dummies(df, columns=['created_month'], prefix='created_month', dtype=int)

    # 2. Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number, 'bool']).copy()
    
    # 3. Fill missing values with 0
    df_numeric = df_numeric.fillna(0)
    
    return df_numeric.values

def apply_target_encoding(X_train, X_val, X_test, y_train, cols_to_encode, m):

    # Create copies so we don't accidentally overwrite original dataframes
    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy()
    
    # Calculate the global average of y_train (used as a fallback)
    global_mean = y_train.mean()
    
    for col in cols_to_encode:
        # 1. Calculate the mean of y_train for each category in this column
        # Example: 'Pothole' -> 4.5 (log hours)
        train_col = X_train[col].astype(str)
        stats = pd.Series(y_train).groupby(train_col.values).agg(['mean', 'count'])
        smoothed_means = (stats['mean'] * stats['count'] + global_mean * m) / (stats['count'] + m)

        X_train_encoded[col] = train_col.map(smoothed_means)
        X_val_encoded[col] = X_val[col].astype(str).map(smoothed_means).fillna(global_mean)
        X_test_encoded[col] = X_test[col].astype(str).map(smoothed_means).fillna(global_mean)
        
    return X_train_encoded, X_val_encoded, X_test_encoded

def load_fold_data(processed_dir, tag, fold_idx):
    X_full = pd.read_parquet(processed_dir / f"Xraw_{tag}.parquet")
    y_full = np.load(processed_dir / f"y_{tag}.npy")
    folds = np.load(processed_dir / f"folds_{tag}.npz")
    
    train_idx, test_idx = folds[f"train_idx_{fold_idx}"], folds[f"test_idx_{fold_idx}"]
    return X_full.iloc[train_idx], X_full.iloc[test_idx], y_full[train_idx], y_full[test_idx]


def main():
    PROCESSED_DIR = Path("data/processed")
    TAG = "processed_2026-03-24_cv5"
    
    print(f"--- Training XGBoost ---")
    
    all_mae = []
    target_encode_cols = ['complaint_type', 'descriptor', 'location_type', 'community_board', 'police_precinct', 'incident_zip', 'council_district']

    for i in range(5):
        # 1. Load
        x_train_full, x_test, y_train_full, y_test = load_fold_data(PROCESSED_DIR, TAG, i)

        x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)
    
        x_train, x_val, x_test = apply_target_encoding(x_train, x_val, x_test, y_train, target_encode_cols, m=15)

        # 2. Clean to Numpy
        X_train = prepare_numpy_data(x_train)
        X_val = prepare_numpy_data(x_val)
        X_test = prepare_numpy_data(x_test)
        
        # 3. Train
        model = MyExtremeBooster(n_estimators=100, lr=.5, patience=5)
        model.fit(X_train, y_train, X_val, y_val)
        
        # 4. Predict & Evaluate
        preds_log = model.predict(X_test)
        
        # Convert back to actual hours
        # actual_hours = np.expm1(y_test)
        # predicted_hours = np.expm1(preds_log)
        # mae = mean_absolute_error(actual_hours, predicted_hours)
        # rmse = np.sqrt(mean_squared_error(actual_hours, predicted_hours))
        # all_mae.append(mae)

        mae = mean_absolute_error(y_test, preds_log)
        rmse = np.sqrt(mean_squared_error(y_test, preds_log))
        all_mae.append(mae)
        
        #errors = actual_hours - predicted_hours

        errors = y_test - preds_log

        
        print(f"Fold {i} | MAE: {mae:.2f} hrs")
        print(f"Fold {i} | RMSE: {rmse:.2f} hrs")
        print("-" * 30)

    print(f"\nFinal Result: Average Error = {np.mean(all_mae):.2f} hours")

if __name__ == "__main__":
    main()