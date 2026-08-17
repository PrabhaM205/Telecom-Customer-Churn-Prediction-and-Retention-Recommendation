from src.data.loader import load_raw_data
from src.data.cleaner import clean_data
from src.data.feature_engineering import engineer_features
from src.preprocessing.pipeline import load_transform

import joblib
import os


print("========== PREDICTION TEST ==========")


# --------------------------------------------------
# 1. Load RAW data
# --------------------------------------------------

raw_df = load_raw_data()

print("\n1. Raw customer data loaded successfully")
print("Shape:", raw_df.shape)


# --------------------------------------------------
# 2. Ask for Customer ID at runtime
# --------------------------------------------------

customer_id = input(
    "\nEnter Customer ID to test: "
).strip()


# --------------------------------------------------
# 3. Find customer in RAW data
# --------------------------------------------------

if "CustomerID" not in raw_df.columns:
    raise AssertionError(
        "'CustomerID' column not found in raw dataset."
    )


customer_raw = raw_df[
    raw_df["CustomerID"].astype(str).str.strip()
    == customer_id
].copy()


if customer_raw.empty:
    print(
        f"\nCustomer ID '{customer_id}' was not found."
    )
    raise SystemExit(1)


print("\n2. Customer found successfully")
print("Customer ID:", customer_id)


# --------------------------------------------------
# 4. Clean customer data
# --------------------------------------------------

customer = clean_data(customer_raw)

print("\n3. Customer cleaning successful")
print("Shape:", customer.shape)


# --------------------------------------------------
# 5. Feature engineering
# --------------------------------------------------

customer = engineer_features(customer)

print("\n4. Customer feature engineering successful")
print("Shape:", customer.shape)


# --------------------------------------------------
# 6. Load trained XGBoost model
# --------------------------------------------------

MODEL_PATH = os.path.join(
    "artifacts",
    "xgboost_churn_model.joblib"
)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )


model = joblib.load(MODEL_PATH)

print("\n5. XGBoost model loaded successfully")


# --------------------------------------------------
# 7. Preprocess customer
# --------------------------------------------------

X_customer = load_transform(customer)

print("\n6. Customer preprocessing successful")
print("Feature shape:", X_customer.shape)


# --------------------------------------------------
# 8. Generate prediction
# --------------------------------------------------

probability = model.predict_proba(
    X_customer
)[:, 1][0]

prediction = model.predict(
    X_customer
)[0]


# --------------------------------------------------
# 9. Display prediction
# --------------------------------------------------

print("\n========== CUSTOMER PREDICTION ==========")

print(f"Customer ID      : {customer_id}")
print(f"Churn Probability: {probability:.2%}")


if prediction == 1:
    print("Prediction       : Likely to Churn")
else:
    print("Prediction       : Likely to Stay")


# --------------------------------------------------
# 10. Validate prediction
# --------------------------------------------------

if not 0 <= probability <= 1:
    raise AssertionError(
        "Churn probability is outside the valid 0-1 range."
    )


print("\n==========================================")
print("PREDICTION TEST PASSED")
print("==========================================")