import pandas as pd


REQUIRED_COLUMNS = {"text", "label"}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_schema(df)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty.")
