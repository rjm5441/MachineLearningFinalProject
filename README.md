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

## How to Run (Typical)
1) Create a frozen dataset snapshot:
   python -m code.data.fetch_311 --limit 200000

2) Build interim + processed datasets (placeholders to be implemented later):
   python -m code.data.make_dataset
   python -m code.features.build_features

3) Train/evaluate models (placeholders to be implemented later):
   python -m code.models.train_baseline
   python -m code.models.train_linear
   python -m code.models.train_tree
   python -m code.models.evaluate
