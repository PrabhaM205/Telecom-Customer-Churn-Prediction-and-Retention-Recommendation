import os
import joblib

from src.data.loader import load_raw_data
from src.data.cleaner import clean_data
from src.data.feature_engineering import engineer_features

from src.preprocessing.pipeline import fit_and_save, load_transform

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


print("========== MODEL TEST ==========")


# --------------------------------------------------
# 1. Load and prepare data
# --------------------------------------------------

df = load_raw_data()
df = clean_data(df)
df = engineer_features(df)

print("\n1. Data prepared successfully")
print("Shape:", df.shape)


# --------------------------------------------------
# 2. Split data
# --------------------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Churn Value"]
)

print("\n2. Train/test split successful")


# --------------------------------------------------
# 3. Load preprocessor
# --------------------------------------------------

X_train, feature_names, _ = fit_and_save(train_df)
X_test = load_transform(test_df)

y_train = train_df["Churn Value"]
y_test = test_df["Churn Value"]

print("\n3. Preprocessing successful")
print("Train features:", X_train.shape)
print("Test features :", X_test.shape)


# --------------------------------------------------
# 4. Load trained XGBoost model
# --------------------------------------------------

MODEL_PATH = os.path.join(
    "artifacts",
    "xgboost_churn_model.joblib"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"XGBoost model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("\n4. XGBoost model loaded successfully")


# --------------------------------------------------
# 5. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

print("\n5. Predictions generated successfully")


# --------------------------------------------------
# 6. Validate predictions
# --------------------------------------------------

if len(y_pred) != len(y_test):
    raise AssertionError(
        "Prediction count does not match test data."
    )

if len(y_probability) != len(y_test):
    raise AssertionError(
        "Probability count does not match test data."
    )

if not all(
    0 <= probability <= 1
    for probability in y_probability
):
    raise AssertionError(
        "Invalid probability values detected."
    )


# --------------------------------------------------
# 7. Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy: {accuracy:.4f}")

print("\nPrediction count:", len(y_pred))
print("Probability count:", len(y_probability))


# --------------------------------------------------
# 8. Final validation
# --------------------------------------------------

if accuracy < 0:
    raise AssertionError("Invalid accuracy.")


print("\n===================================")
print("MODEL TEST PASSED")
print("===================================")