from src.data.loader import load_raw_data
from src.data.cleaner import clean_data
from src.data.feature_engineering import engineer_features


print("========== DATA PIPELINE TEST ==========")


# 1. Test data loading
df = load_raw_data()

print("\n1. Raw data loaded successfully")
print("Shape:", df.shape)


# 2. Test cleaning
cleaned_df = clean_data(df)

print("\n2. Data cleaning successful")
print("Shape:", cleaned_df.shape)


# 3. Test feature engineering
featured_df = engineer_features(cleaned_df)

print("\n3. Feature engineering successful")
print("Shape:", featured_df.shape)


# 4. Basic validation
if featured_df.empty:
    raise AssertionError("Feature-engineered dataset is empty.")

if "Churn Value" not in featured_df.columns:
    raise AssertionError(
        "'Churn Value' column is missing."
    )


print("\n4. Basic validation successful")
print("Churn Value column found.")


print("\n========================================")
print("DATA PIPELINE TEST PASSED")
print("========================================")