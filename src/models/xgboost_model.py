import os
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from src.data.loader import load_raw_data
from src.data.cleaner import clean_data
from src.data.feature_engineering import engineer_features
from src.preprocessing.pipeline import fit_and_save, load_transform


# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "artifacts",
    "xgboost_churn_model.joblib"
)


# --------------------------------------------------
# Train XGBoost
# --------------------------------------------------

def train_xgboost():

    # 1. Load raw data
    df = load_raw_data()

    # 2. Clean data
    df = clean_data(df)

    # 3. Feature engineering
    df = engineer_features(df)

    # 4. Train / Test split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Churn Value"]
    )

    # 5. Preprocess training data
    X_train, feature_names, _ = fit_and_save(train_df)

    # 6. Preprocess test data
    X_test = load_transform(test_df)

    # 7. Target
    y_train = train_df["Churn Value"]
    y_test = test_df["Churn Value"]

    # --------------------------------------------------
    # 8. Create XGBoost model
    # --------------------------------------------------

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    # --------------------------------------------------
    # 9. Train model
    # --------------------------------------------------

    print("Training XGBoost model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed!")

    # --------------------------------------------------
    # 10. Prediction
    # --------------------------------------------------

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # 11. Evaluation
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)

    print("\n========== XGBoost Results ==========")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --------------------------------------------------
    # 12. Save model
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved at:")
    print(MODEL_PATH)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    train_xgboost()