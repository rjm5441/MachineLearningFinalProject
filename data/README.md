# data/

## Purpose
This folder holds raw snapshots, intermediate cleaned outputs, processed model-ready tables, and saved models.

## Subfolders
- raw/: frozen snapshots pulled from NYC Open Data (CSV + metadata JSON)
- interim/: cleaned tables before final feature engineering
- processed/: final feature matrix/target ready for modeling
- models/: serialized trained models (joblib/pkl)

## Reproducibility Rule
All model comparisons must use the same raw snapshot (data/raw/*.csv) recorded in the snapshot metadata.
