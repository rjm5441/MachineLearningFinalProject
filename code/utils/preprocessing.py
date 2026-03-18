# code/utils/preprocessing.py
from __future__ import annotations # for future dataclass features

from typing import List     # for type hints of list of column names

from sklearn.compose import ColumnTransformer   # for applying different preprocessing to different columns
from sklearn.impute import SimpleImputer        # for handling missing values in both numeric and categorical features
from sklearn.pipeline import Pipeline       # for chaining preprocessing steps together (e.g., imputation + encoding/scaling)   
from sklearn.preprocessing import OneHotEncoder, StandardScaler     # for encoding categorical features and scaling numeric features, respectively

'''
Utility function to create a ColumnTransformer for preprocessing features.
This is used in the model training scripts to ensure consistent preprocessing across models.
'''
def make_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    """
    Returns an unfitted ColumnTransformer that:
      - imputes + one-hot encodes categoricals
      - imputes + standardizes numerics
    Fit this on TRAIN only inside each CV fold.
    """
    # Define separate pipelines for categorical and numeric features, then combine them into a ColumnTransformer.
    # The categorical pipeline imputes missing values using the most frequent value and then applies one-hot encoding, 
    # while the numeric pipeline imputes missing values using the median and then standardizes the features using z-score normalization.
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # sparse_threshold=0.3 means if the output of the ColumnTransformer is more than 30% sparse, 
    # it will return a sparse matrix instead of a dense one, 
    # which can save memory when there are many categorical features after one-hot encoding.
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_cols),
            ("num", num_pipe, numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )