"""
src/data/loader.py

Single data-access boundary for the project (see documentation Sec. 3 / Sec. 14).
Every other module reads customer data through load_raw_data() — if the source
ever moves from a local Excel file to a database or API, this is the only
file that needs to change.
"""

from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "Telco_customer_churn.xlsx"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Telco customer churn dataset exactly as provided.
    No cleaning or transformation happens here — that belongs in cleaner.py.

    Returns
    -------
    pd.DataFrame with 7,043 rows and 33 columns (IBM Watson Analytics schema).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {path}. "
            "Place Telco_customer_churn.xlsx under data/raw/."
        )
    df = pd.read_excel(path)
    return df


def basic_schema_check(df: pd.DataFrame) -> None:
    """
    Fast fail-early check — run this right after loading, before any
    downstream module touches the data. Catches upstream schema drift
    (e.g. a column renamed or removed in a future data refresh).
    """
    required_columns = {
        "CustomerID", "Gender", "Senior Citizen", "Partner", "Dependents",
        "Tenure Months", "Phone Service", "Internet Service", "Contract",
        "Payment Method", "Monthly Charges", "Total Charges",
        "Churn Label", "Churn Value",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns missing from raw data: {missing}")


if __name__ == "__main__":
    data = load_raw_data()
    basic_schema_check(data)
    print(f"Loaded {data.shape[0]} rows, {data.shape[1]} columns.")
    print(data.head())