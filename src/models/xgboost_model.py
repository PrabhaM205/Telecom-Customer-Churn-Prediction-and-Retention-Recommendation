import os
import json
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

from src.preprocessing.pipeline import (
    fit_and_save,
    load_transform
)
from src.config import get

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

ARTIFACTS_DIR = os.path.join(
    PROJECT_ROOT,
    "artifacts"
)

MODEL_PATH = os.path.join(
    ARTIFACTS_DIR,
    "xgboost_churn_model.joblib"
)

METRICS_PATH = os.path.join(
    ARTIFACTS_DIR,
    "xgboost_metrics.json"
)


# ============================================================
# Decision Threshold
# ============================================================

THRESHOLD = get("model", "xgboost", "decision_threshold", default=0.55)


# ============================================================
# Train XGBoost
# ============================================================

def train_xgboost():

    print("\n==============================================")
    print("          XGBOOST TRAINING PIPELINE")
    print("==============================================")


    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    print("\n[1] Loading raw data...")

    df = load_raw_data()

    print(f"Raw data shape: {df.shape}")


    # --------------------------------------------------------
    # 2. Clean data
    # --------------------------------------------------------

    print("\n[2] Cleaning data...")

    df = clean_data(df)

    print(f"Cleaned data shape: {df.shape}")


    # --------------------------------------------------------
    # 3. Feature engineering
    # --------------------------------------------------------

    print("\n[3] Feature engineering...")

    df = engineer_features(df)

    print(f"Feature engineered shape: {df.shape}")


    # --------------------------------------------------------
    # 4. Train / Test split
    # --------------------------------------------------------

    print("\n[4] Splitting train and test data...")

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["Churn Value"]
    )

    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples : {len(test_df)}")


    # --------------------------------------------------------
    # 5. Preprocessing
    # --------------------------------------------------------

    print("\n[5] Preprocessing training data...")

    X_train, feature_names, _ = fit_and_save(
        train_df
    )

    print(
        f"Output shape: {X_train.shape} | "
        f"Features: {len(feature_names)}"
    )


    print("\nPreprocessing test data...")

    X_test = load_transform(
        test_df
    )

    print(f"Test output shape: {X_test.shape}")


    # --------------------------------------------------------
    # 6. Target
    # --------------------------------------------------------

    print("\n[6] Preparing target variable...")

    y_train = train_df["Churn Value"]
    y_test = test_df["Churn Value"]

    print(
        f"Train churned: {int(y_train.sum())}"
    )

    print(
        f"Test churned : {int(y_test.sum())}"
    )


    # --------------------------------------------------------
    # 7. Class imbalance
    # --------------------------------------------------------

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        f"\nscale_pos_weight: "
        f"{scale_pos_weight:.3f}"
    )


    # --------------------------------------------------------
    # 8. Create XGBoost model
    # --------------------------------------------------------

    print("\n[7] Creating XGBoost model...")

    model = XGBClassifier(

        n_estimators=300,

        max_depth=3,

        learning_rate=0.03,

        subsample=0.6,

        colsample_bytree=0.6,

        min_child_weight=5,

        gamma=0.3,

        reg_alpha=1,

        reg_lambda=2,

        scale_pos_weight=scale_pos_weight,

        random_state=42,

        eval_metric="logloss"
    )


    # --------------------------------------------------------
    # 9. Train
    # --------------------------------------------------------

    print("\nTraining XGBoost model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed!")


    # --------------------------------------------------------
    # 10. Prediction
    # --------------------------------------------------------

    print("\n[8] Making predictions...")

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    y_pred = (
        y_probability >= THRESHOLD
    ).astype(int)


    # --------------------------------------------------------
    # 11. Evaluation
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


    # --------------------------------------------------------
    # 12. Print results
    # --------------------------------------------------------

    print("\n==============================================")
    print("             XGBOOST RESULTS")
    print("==============================================")

    print(f"Threshold : {THRESHOLD}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


    # --------------------------------------------------------
    # 13. Save metrics
    # --------------------------------------------------------

    os.makedirs(
        ARTIFACTS_DIR,
        exist_ok=True
    )

    metrics = {

        "model": "XGBoost",

        "threshold": THRESHOLD,

        "accuracy": round(
            float(accuracy),
            4
        ),

        "precision": round(
            float(precision),
            4
        ),

        "recall": round(
            float(recall),
            4
        ),

        "f1_score": round(
            float(f1),
            4
        ),

        "roc_auc": round(
            float(roc_auc),
            4
        ),

        "scale_pos_weight": round(
            float(scale_pos_weight),
            4
        ),

        "n_train_samples": int(
            len(y_train)
        ),

        "n_test_samples": int(
            len(y_test)
        ),

        "n_features": int(
            len(feature_names)
        )
    }


    with open(
        METRICS_PATH,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )


    print("\nMetrics saved at:")
    print(METRICS_PATH)


    # --------------------------------------------------------
    # 14. Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModel saved at:")
    print(MODEL_PATH)


    print("\n==============================================")
    print("       XGBOOST TRAINING COMPLETED")
    print("==============================================")


# ============================================================
# Direct execution
# ============================================================

if __name__ == "__main__":
    train_xgboost()