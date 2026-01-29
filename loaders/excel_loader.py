import pandas as pd
from io import BytesIO

def load_excel(excel_data: bytes, file_name: str = "", markdown: bool = False) -> str:
    """Extract text from Excel or CSV file and optionally return markdown."""
    print("Excel parser invoked")
    try:
        all_text = ""
        excel_file = BytesIO(excel_data)

        # Handle Excel files (.xls, .xlsx, .ods)
        with pd.ExcelFile(excel_file) as xls:
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
                df.fillna("", inplace=True)

                if markdown:
                    headers = " | ".join(df.columns)
                    separator = " | ".join(["---"] * len(df.columns))
                    rows = [" | ".join(map(str, row)) for row in df.values]
                    table = "\n".join([headers, separator] + rows)
                    all_text += f"\n### Sheet: {sheet_name}\n\n{table}\n"
                else:
                    sheet_text = df.to_string(index=False)
                    all_text += f"Sheet: {sheet_name}\n{sheet_text}\n"
        # print("all_text:", all_text)
        return all_text
    except Exception as e:
        raise Exception(f"Error processing Excel/CSV file: {e}")
