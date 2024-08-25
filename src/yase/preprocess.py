import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame (e.g., handle missing values)."""
    df = df.dropna()  # Example: drop rows with missing values
    return df
