import os
import pandas as pd

def test_dataset_columns():
    path = "data/processed/image_report_pairs.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        assert "image_id" in df.columns
        assert "image_path" in df.columns
        assert "report" in df.columns
