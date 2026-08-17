import os
import json
import joblib
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.data.cleaner import clean_data
from src.data.feature_engineering import engineer_features

from src.preprocessing.pipeline import fit_and_save, load_transform


# ============================================================
# PROJECT PATHS
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
    "xgboost_retraining_metrics.json"
)


# ============================================================
# AUTOMATIC RETRAINING
# ============================================================

def retrain_from_incoming_data(incoming_df):

    print("\n")
    print("=" * 60)
    print("AUTOMATIC XGBOOST RETRAINING")
    print("=" * 60)

    print(
        f"\nIncoming data shape: {incoming_df.shape}"
    )

    # --------------------------------------------------------
    # 1. Validate incoming data
    # --------------------------------------------------------

    if incoming_df.empty:

        raise ValueError(
            "Incoming data cannot be empty."
        )

    if "Churn Value" not in incoming_df.columns:

        raise ValueError(
            "'Churn Value' column is required "
            "for supervised retraining."
        )

    # --------------------------------------------------------
    # 2. Clean data
    # --------------------------------------------------------

    print("\n[1] Cleaning incoming data...")

    df = clean_data(
        incoming_df.copy()
    )

    print(
        f"Cleaned data shape: {df.shape}"
    )

    # --------------------------------------------------------
    # 3. Feature engineering
    # --------------------------------------------------------

    print("\n[2] Feature engineering...")

    df = engineer_features(df)

    print(
        f"Feature engineered shape: {df.shape}"
    )

    # --------------------------------------------------------
    # 4. Train/test split
    # --------------------------------------------------------

    print("\n[3] Splitting incoming data...")

    train_df, test_df = train_test_split(

        df,

        test_size=0.20,

        random_state=42,

        stratify=df["Churn Value"]
    )

    print(
        f"Training samples: {len(train_df)}"
    )

    print(
        f"Testing samples : {len(test_df)}"
    )

    # --------------------------------------------------------
    # 5. Fit preprocessing
    # --------------------------------------------------------

    print(
        "\n[4] Fitting preprocessing pipeline..."
    )

    X_train, feature_names, _ = fit_and_save(
        train_df
    )

    X_test = load_transform(
        test_df
    )

    print(
        f"Training feature shape: "
        f"{X_train.shape}"
    )

    print(
        f"Testing feature shape : "
        f"{X_test.shape}"
    )

    # --------------------------------------------------------
    # 6. Target
    # --------------------------------------------------------

    y_train = train_df[
        "Churn Value"
    ]

    y_test = test_df[
        "Churn Value"
    ]

    # --------------------------------------------------------
    # 7. Class imbalance
    # --------------------------------------------------------

    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    if positive_count == 0:

        raise ValueError(
            "Incoming data contains no churned "
            "customers. Retraining cannot continue."
        )

    scale_pos_weight = (
        negative_count /
        positive_count
    )

    print(
        f"\nScale positive weight: "
        f"{scale_pos_weight:.3f}"
    )

    # --------------------------------------------------------
    # 8. Create XGBoost model
    # --------------------------------------------------------

    print(
        "\n[5] Creating XGBoost model..."
    )

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
    # 9. Train model
    # --------------------------------------------------------

    print(
        "\n[6] Training model..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Incoming-data retraining completed."
    )

    # --------------------------------------------------------
    # 10. Prediction
    # --------------------------------------------------------

    print(
        "\n[7] Evaluating retrained model..."
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.55
    ).astype(int)

    # --------------------------------------------------------
    # 11. Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    # --------------------------------------------------------
    # 12. Display metrics
    # --------------------------------------------------------

    print("\n")
    print("=" * 46)
    print("       RETRAINED MODEL RESULTS")
    print("=" * 46)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
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

        "retrained": True,

        "reason": "data_drift",

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

        "training_samples": int(
            len(y_train)
        ),

        "testing_samples": int(
            len(y_test)
        ),

        "features": int(
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

    # --------------------------------------------------------
    # 14. Save new model
    # --------------------------------------------------------

    print(
        "\n[8] Saving retrained model..."
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        "\nNew model saved at:"
    )

    print(
        MODEL_PATH
    )

    print("\n")
    print("=" * 60)
    print(
        "AUTOMATIC RETRAINING COMPLETED"
    )
    print("=" * 60)

    return {

        "model": model,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "roc_auc": roc_auc,

        "model_path": MODEL_PATH,

        "metrics_path": METRICS_PATH
    }