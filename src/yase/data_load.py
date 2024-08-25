import pandas as pd
import json

def load_csv(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def load_json(file_path: str) -> dict:
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)
