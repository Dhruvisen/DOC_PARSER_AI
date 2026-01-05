import pandas as pd

def load_excel(path: str) -> str:
    df = pd.read_excel(path)
    return df.to_string(index=False)
