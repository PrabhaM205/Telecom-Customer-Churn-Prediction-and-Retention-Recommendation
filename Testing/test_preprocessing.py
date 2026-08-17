from src.data.loader import load_raw_data
from src.data.cleaner import clean_data
from src.data.feature_engineering import engineer_features

from src.preprocessing.pipeline import (
    fit_and_save,
    load_transform
)

from sklearn.model_selection import train_test_split


print("========== PREPROCESSING TEST ==========")


# 1. Load and prepare data
df = load_raw_data()
df = clean_data(df)
df = engineer_features(df)

print("\n1. Data prepared successfully")
print("Shape:", df.shape)


# 2. Split data
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Churn Value"]
)

print("\n2. Train/test split successful")
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)


# 3. Fit preprocessor on training data
X_train, feature_names, preprocessor = fit_and_save(
    train_df
)

print("\n3. Preprocessor fitted successfully")
print("Transformed train shape:", X_train.shape)
print("Number of features:", len(feature_names))


# 4. Transform test data using saved preprocessor
X_test = load_transform(test_df)

print("\n4. Test transformation successful")
print("Transformed test shape:", X_test.shape)


# 5. Validate shapes
if X_train.shape[1] != X_test.shape[1]:
    raise AssertionError(
        "Train and test feature counts do not match."
    )


if X_train.shape[1] != len(feature_names):
    raise AssertionError(
        "Feature name count does not match transformed data."
    )


print("\n5. Preprocessing validation successful")


print("\n========================================")
print("PREPROCESSING TEST PASSED")
print("========================================")