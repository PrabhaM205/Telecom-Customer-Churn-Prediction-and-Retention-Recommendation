from src.data.loader import load_raw_data
from src.models.retrain import retrain_xgboost


print("=" * 60)
print("RETRAINING TEST")
print("=" * 60)


# --------------------------------------------------
# 1. Load existing dataset
# --------------------------------------------------

print("\n1. Loading training data...")

df = load_raw_data()

print(
    f"Dataset shape: {df.shape}"
)


# --------------------------------------------------
# 2. Use a sample as incoming data
# --------------------------------------------------

print("\n2. Creating incoming-data sample...")

incoming_df = df.sample(
    n=1000,
    random_state=42
).copy()

print(
    f"Incoming data shape: "
    f"{incoming_df.shape}"
)


# --------------------------------------------------
# 3. Retrain
# --------------------------------------------------

print("\n3. Starting retraining...")

result = retrain_xgboost(
    incoming_df
)


# --------------------------------------------------
# 4. Validate
# --------------------------------------------------

print("\n4. Validating retraining result...")

assert result["model_path"]

assert result["metrics_path"]

assert 0 <= result["accuracy"] <= 1

assert 0 <= result["precision"] <= 1

assert 0 <= result["recall"] <= 1

assert 0 <= result["f1_score"] <= 1

assert 0 <= result["roc_auc"] <= 1


print(
    "\nRetraining validation successful"
)


print("\n" + "=" * 60)
print("RETRAINING TEST PASSED")
print("=" * 60)