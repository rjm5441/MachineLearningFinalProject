from pathlib import Path

# Dataset configuration
DATASET_ID = "erm2-nwe9"  # NYC Open Data: 311 Service Requests from 2020 to Present
SOCRATA_DOMAIN = "data.cityofnewyork.us"
SOCRATA_RESOURCE = f"https://{SOCRATA_DOMAIN}/resource/{DATASET_ID}.csv"

# Local paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Reproducibility
RANDOM_SEED = 42
