import pandas as pd
from io import BytesIO

def load_csv(csv_data: bytes, markdown: bool = False) -> str:
    """Extract text from CSV data."""
    try:
        csv_file = BytesIO(csv_data)
        df = pd.read_csv(csv_file, dtype=str)
        df.fillna("", inplace=True)
        
        if markdown:
            headers = " | ".join(df.columns)
            separator = " | ".join(["---"] * len(df.columns))
            rows = [" | ".join(map(str, row)) for row in df.values]
            table = "\n".join([headers, separator] + rows)
            return f"\n### CSV Data\n\n{table}\n"
        else:
            return df.to_string(index=False)
            
    except Exception as e:
        raise Exception(f"Error processing CSV file: {e}")
