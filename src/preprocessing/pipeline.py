# src/preprocessing/pipeline.py

import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Column groups — single source of truth. Update HERE if feature_engineering.py
# adds/removes a column; nothing downstream needs to change.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
    "ARPU",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Tenure Bucket",
]

# Already 0/1 from feature_engineering.py — pass through untouched, no encoding needed
BINARY_PASSTHROUGH_FEATURES = [
    "Protective Bundle",
    "High Risk Contract",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_PASSTHROUGH_FEATURES

DEFAULT_ARTIFACT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "artifacts", "preprocessor.joblib"
)


def build_preprocessor() -> ColumnTransformer:
    """
    Builds the ColumnTransformer. Called once at training time; the FITTED
    object is what gets saved and reused at inference — never rebuild fresh
    at inference time, or you risk encoder categories drifting from training.
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_PASSTHROUGH_FEATURES),
        ],
        remainder="drop",  # explicit: anything not listed above (IDs, leakage cols) is dropped
        verbose_feature_names_out=False,
    )
    return preprocessor


def fit_and_save(
    df: pd.DataFrame,
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
) -> tuple:
    """
    Fits the preprocessor on training data and persists it to disk.
    Returns (transformed_array, feature_names, fitted_preprocessor).

    Call this ONCE, at training time, on the training split only
    (never on the full dataset — that would leak test-set distribution
    into the scaler/encoder).
    """
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns for preprocessing: {missing}")

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(df[ALL_FEATURES])
    feature_names = preprocessor.get_feature_names_out()

    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(preprocessor, artifact_path)
    print(f"Fitted preprocessor saved: {artifact_path}")
    print(f"Output shape: {X_transformed.shape}  |  Features: {len(feature_names)}")

    return X_transformed, feature_names, preprocessor


def load_transform(
    df: pd.DataFrame,
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
):
    """
    Loads the FITTED preprocessor and applies it to new data
    (validation/test split, or a single customer at inference time).
    This is the function src/prediction/predictor.py should call —
    guarantees identical encoding/scaling as training, every time.
    """
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"No fitted preprocessor found at {artifact_path}. "
            f"Run fit_and_save() on training data first."
        )

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns for preprocessing: {missing}")

    preprocessor = joblib.load(artifact_path)
    X_transformed = preprocessor.transform(df[ALL_FEATURES])
    return X_transformed


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
    from feature_engineering import engineer_features
    from cleaner import clean_data
    from loader import load_raw_data
    from sklearn.model_selection import train_test_split

    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)

    # Split BEFORE fitting the preprocessor — fit only on train
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Churn Value"]
    )

    X_train, feature_names, _ = fit_and_save(train_df)
    X_test = load_transform(test_df)

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Sample feature names: {list(feature_names[:10])}")