# src/data/feature_engineering.py

import pandas as pd
import numpy as np


def add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket Tenure Months into lifecycle stages."""
    bins = [-1, 6, 12, 24, 48, np.inf]
    labels = ["0-6mo", "7-12mo", "13-24mo", "25-48mo", "48mo+"]
    df["Tenure Bucket"] = pd.cut(df["Tenure Months"], bins=bins, labels=labels)
    return df


def add_arpu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average Revenue Per User (per month).
    Cleaner CLV proxy than raw Total Charges, which is confounded by tenure
    (a long-tenure low-value customer can have a higher Total Charges than
    a short-tenure high-value one).
    """
    # Guard against tenure = 0 (brand-new customers, already surfaced in EDA step 2)
    df["ARPU"] = np.where(
        df["Tenure Months"] > 0,
        df["Total Charges"].astype(float) / df["Tenure Months"],
        df["Monthly Charges"]  # fall back to current monthly rate for tenure=0 rows
    )
    return df


def add_protective_bundle_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags customers with BOTH Online Security and Tech Support.
    From your EDA: customers lacking these two services show ~42% and ~41.6%
    churn respectively vs ~15% when subscribed — a strong combined signal.
    """
    df["Protective Bundle"] = (
        (df["Online Security"] == "Yes") & (df["Tech Support"] == "Yes")
    ).astype(int)
    return df


def add_contract_risk_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags the highest-risk combination surfaced in EDA:
    Month-to-month contract + Electronic check payment.
    (42.7% churn for month-to-month, 45.3% for electronic check individually —
    this flag captures customers hit by both.)
    """
    df["High Risk Contract"] = (
        (df["Contract"] == "Month-to-month") &
        (df["Payment Method"] == "Electronic check")
    ).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single entry point — chain all derived features here."""
    df = df.copy()
    df = add_tenure_bucket(df)
    df = add_arpu(df)
    df = add_protective_bundle_flag(df)
    df = add_contract_risk_flag(df)
    return df


if __name__ == "__main__":
    from cleaner import clean_data      # adjust import to match your actual cleaner.py entry point
    from loader import load_raw_data    # adjust to your actual loader.py entry point

    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)

    print(f"Feature-engineered shape: {df.shape}")
    print(f"New columns added: {['Tenure Bucket', 'ARPU', 'Protective Bundle', 'High Risk Contract']}")
    print(df[["Tenure Bucket", "ARPU", "Protective Bundle", "High Risk Contract"]].head())

    # src/data/feature_engineering.py
# ... (keep everything above as-is) ...

import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def save_processed_data(df: pd.DataFrame, filename: str = "telco_features.csv") -> str:
    """
    Persists the feature-engineered dataset to data/processed/.
    Keeping this as a checkpoint (separate from raw/cleaned) means
    src/preprocessing/pipeline.py and model training scripts can load
    directly from here without re-running cleaning + feature engineering
    every time.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(out_path, index=False)
    print(f"Saved processed dataset: {out_path}  (shape={df.shape})")
    return out_path


if __name__ == "__main__":
    from cleaner import clean_data
    from loader import load_raw_data

    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)

    print(f"Feature-engineered shape: {df.shape}")
    print(f"New columns added: {['Tenure Bucket', 'ARPU', 'Protective Bundle', 'High Risk Contract']}")
    print(df[["Tenure Bucket", "ARPU", "Protective Bundle", "High Risk Contract"]].head())

    save_processed_data(df)