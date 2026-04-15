# Title: Predicting NYC 311 Complaint Resolution Time (2020–Present)

## Abstract
We compare classical machine learning models and more advanced regression approaches to predict the resolution time of NYC 311 complaints using the NYC Open Data 311 Service Requests dataset (2020–present). Models are evaluated on a common, frozen snapshot of the dataset using MAE and RMSE.
The following models are compared: 
   1) XGboost,
   2) Random Forest Regressor, 
   3) Residual Neural Network,
   4) Multilayer Perceptron.

## Developers
- Ryan Miner
- Myra Mulongoti
- Obumneme Umeonwuka

## Repository Structure
- code/: scripts, implementations
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

   - Commands to run training and make predictions:
      - python -m code.models.train_mlp
      - python -m code.models.train_neural_net
      - python -m code.models.train_tree
      - python -m code.models.train_xgb
     
  
