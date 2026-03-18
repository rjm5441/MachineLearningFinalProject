# Feature Build Report — processed_2026-03-13_cv5

##Results
- Raw feature table (not encoded/scaled)
- Target vector aligned to the raw table
- Reproducible CV fold indices

## Source
- Interim file: `MachineLearningFinalProject/data/interim/311_cleaned_2026-03-13.parquet`
- Generated at (UTC): `2026-03-18T00:12:48Z`

## Target
- Column: `resolution_time_hours`
- Transform: `log1p`

## Feature Columns Used (raw)
- Categorical (9): agency, complaint_type, descriptor, borough, location_type, incident_zip, community_board, council_district, police_precinct
- Numeric (2): latitude, longitude
- Time-derived from `created_date` (4): created_hour, created_dayofweek, created_month, created_is_weekend

## Intended Encoding / Scaling (fit per-fold to avoid leakage)
- Categorical: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')
- Numeric + time: SimpleImputer(median) + StandardScaler

## CV Folds
- CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- Stratification: quantile bins of y (requested bins = 10)
- Fold sizes (train, test):
  - Fold 0: (128756, 32189)
  - Fold 1: (128756, 32189)
  - Fold 2: (128756, 32189)
  - Fold 3: (128756, 32189)
  - Fold 4: (128756, 32189)

## Notes
- Preprocessing must be fit on the training split inside each fold.
- All models should use the same folds + same preprocessing definition for fair comparison.
