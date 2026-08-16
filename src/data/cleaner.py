"""
src/data/cleaner.py

Cleans the raw Telco dataset into a model-ready base table.
Does NOT do feature engineering (see feature_engineering.py) — this module
only fixes data-quality issues: types, blanks, redundant/leaky columns.

Design note: kept deliberately separate from feature_engineering.py so
"fixing broken data" and "creating new signal" can be tested and reasoned
about independently (see documentation Sec. 6).
"""

import pandas as pd

# Columns excluded from modeling — either leakage risk or non-predictive.
# Kept as a named constant (not scattered magic strings) so the reasoning
# is visible and easy to revisit.
LEAKAGE_COLUMNS = [
    "Churn Score",   # IBM's own pre-computed churn likelihood — proxy for the label
    "Churn Reason",  # only populated AFTER churn — 100% leakage as a feature
]

REDUNDANT_COLUMNS = [
    "Count",       # constant = 1 for every row
    "Country",     # constant = "United States"
    "State",       # constant = "California"
    "Lat Long",    # duplicates Latitude + Longitude
    "Churn Label", # duplicates Churn Value (Yes/No vs 1/0) — keep numeric target only
]

# City/Zip/Lat/Long are kept in the cleaned table (not dropped) but excluded
# from the default model feature set — they're useful for a future geo-based
# segmentation feature, just too high-cardinality for the Day-1 baseline.
GEO_COLUMNS_NOT_USED_BY_DEFAULT = ["City", "Zip Code", "Latitude", "Longitude"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all data-quality fixes and return a cleaned DataFrame.
    Safe to call multiple times (idempotent) — does not mutate the input.
    """
    df = df.copy()

    # --- 1. Fix Total Charges: stored as text, blank for tenure=0 customers ---
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    # Structural, not random: brand-new customers (tenure=0) have no billing
    # history yet. Impute as tenure * monthly charge (= 0 here), not the
    # column mean, which would invent a false billing history.
    df["Total Charges"] = df["Total Charges"].fillna(
        df["Tenure Months"] * df["Monthly Charges"]
    )

    # --- 2. Standardize Yes/No-style categorical text ---
    # Several service columns use "No internet service" / "No phone service"
    # as a third category rather than a plain "No" — collapse to "No" so
    # downstream encoding treats it as a genuine binary flag, with the
    # underlying "does this customer have internet/phone at all" already
    # captured separately by Internet Service / Phone Service.
    collapse_cols = [
        "Online Security", "Online Backup", "Device Protection",
        "Tech Support", "Streaming TV", "Streaming Movies", "Multiple Lines",
    ]
    for col in collapse_cols:
        df[col] = df[col].replace(
            {"No internet service": "No", "No phone service": "No"}
        )

    # --- 3. Drop leakage and redundant columns ---
    df = df.drop(columns=LEAKAGE_COLUMNS + REDUNDANT_COLUMNS, errors="ignore")

    # --- 4. Type cleanup ---
    df["Senior Citizen"] = df["Senior Citizen"].map({"Yes": 1, "No": 0}).astype(int)

    # --- 5. Duplicate check ---
    n_dupes = df["CustomerID"].duplicated().sum()
    if n_dupes:
        df = df.drop_duplicates(subset="CustomerID", keep="first")

    # --- 6. Sanity assertions — fail loudly if a future data refresh breaks these ---
    assert df["Total Charges"].isna().sum() == 0, "Unresolved nulls in Total Charges"
    # Churn Value only exists at TRAINING time (it's the label). At inference
    # time, predictor.py calls clean_data() on a new customer we're trying
    # to predict FOR — so this column legitimately won't exist. Only
    # validate it when present, so this function works correctly in both
    # training and live-prediction contexts.
    if "Churn Value" in df.columns:
        assert df["Churn Value"].isin([0, 1]).all(), "Churn Value must be binary"
    assert df["Tenure Months"].ge(0).all(), "Negative tenure found"

    return df.reset_index(drop=True)


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Returns the column list that should feed the model — i.e. cleaned
    columns minus identifiers, target, and geo columns not yet used.
    Centralized here so src/preprocessing/pipeline.py and src/models/*
    always agree on what counts as a "feature."
    """
    exclude = {"CustomerID", "Churn Value"} | set(GEO_COLUMNS_NOT_USED_BY_DEFAULT)
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    from loader import load_raw_data

    raw = load_raw_data()
    cleaned = clean_data(raw)
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Model feature columns ({len(get_model_feature_columns(cleaned))}):")
    print(get_model_feature_columns(cleaned))