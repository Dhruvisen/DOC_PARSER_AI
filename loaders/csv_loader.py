import pandas as pd

def load_csv(path: str) -> str:
    df = pd.read_csv(path)
    return df.to_string(index=False)
