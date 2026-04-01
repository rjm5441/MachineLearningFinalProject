'''
Opens the predictions parquet file and prints the first few rows to an inspectable csv file.
'''

import pandas as pd

df = pd.read_parquet("preds_random_forest_regressor_BEST_processed_2026-03-13_cv5.parquet")

# Wrtie the first 50 rows to a csv file
df.head(50).to_csv("preds_random_forest_regressor_BEST_processed_2026-03-13_cv5_head.csv", index=False)
