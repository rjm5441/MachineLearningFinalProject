# Title: Predicting NYC 311 Complaint Resolution Time (2020–Present)

## Abstract
We compare classical machine learning models and more advanced regression approaches to predict the resolution time of NYC 311 complaints using the NYC Open Data 311 Service Requests dataset (2020–present). Models are evaluated on a common, frozen snapshot of the dataset using MAE and RMSE.

## Developers
- Ryan Miner
- Myra Mulongoti
- Obum

## Repository Structure
- code/: scripts, notebooks, implementations
- resources/: papers, links, notes, slides
- data/: datasets, snapshots, processed tables, saved models
- reports/: figures and tables exported for writeups/slides

## How to Run
### 1) Create a frozen dataset snapshot:
   python -m code.data.fetch_311 --limit 200000

### 2) Build interim cleaned dataset:
   python -m code.data.make_dataset

### 3) Build final feature inputs + CV folds:
   python -m code.features.build_features

### 4) Train/evaluate models:

   All model training scripts:
   - load `Xraw_*`, `y_*`, `folds_*`, and `feature_spec_*` from `data/processed/`
   - for each CV fold:
      - fit the preprocessing pipeline only on the training split
      - transform train/test split
      - train the model and compute metrics 
      - write per-fold + average results to `reports/tables/` (and/or print to console)


   - Run commands:
      - python -m code.models.train_baseline
      - python -m code.models.train_linear
      - python -m code.models.train_tree
      etc...
      - python -m code.models.evaluate
  
