import json
import os

from src.models.model_manager import ModelManager


print("=" * 60)
print("INITIAL MODEL REGISTRATION")
print("=" * 60)


# ============================================================
# 1. Create Model Manager
# ============================================================

manager = ModelManager()


# ============================================================
# 2. Check whether a production model already exists
# ============================================================

current_version = manager.get_current_version()

if current_version is not None:

    print(
        f"\nProduction model already exists: "
        f"{current_version}"
    )

    print(
        "\nNo new registration required."
    )

    manager.display_history()

    raise SystemExit(0)


# ============================================================
# 3. Model path
# ============================================================

model_path = os.path.join(
    manager.artifacts_dir,
    "xgboost_churn_model.joblib"
)


if not os.path.exists(model_path):

    raise FileNotFoundError(
        f"\nProduction model not found:\n"
        f"{model_path}"
    )


print(
    "\nProduction model found:"
)

print(model_path)


# ============================================================
# 4. Load metrics
# ============================================================

metrics_path = os.path.join(
    manager.artifacts_dir,
    "xgboost_metrics.json"
)


if not os.path.exists(metrics_path):

    raise FileNotFoundError(
        f"\nMetrics file not found:\n"
        f"{metrics_path}"
    )


with open(
    metrics_path,
    "r"
) as file:

    metrics = json.load(file)


print(
    "\nExisting model metrics:"
)

print(
    f"Accuracy  : {metrics['accuracy']}"
)

print(
    f"Precision : {metrics['precision']}"
)

print(
    f"Recall    : {metrics['recall']}"
)

print(
    f"F1 Score  : {metrics['f1_score']}"
)

print(
    f"ROC-AUC   : {metrics['roc_auc']}"
)


# ============================================================
# 5. Register as v1
# ============================================================

version = manager.get_next_version()

print(
    f"\nRegistering existing model as "
    f"{version}..."
)


manager.register_model(

    version=version,

    accuracy=metrics["accuracy"],

    precision=metrics["precision"],

    recall=metrics["recall"],

    f1_score=metrics["f1_score"],

    roc_auc=metrics["roc_auc"],

    status="production"
)


# ============================================================
# 6. Display result
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "INITIAL MODEL REGISTERED SUCCESSFULLY"
)

print(
    "=" * 60
)

print(
    f"\nProduction version: "
    f"{manager.get_current_version()}"
)

print(
    f"Accuracy          : "
    f"{metrics['accuracy']}"
)

print(
    f"ROC-AUC           : "
    f"{metrics['roc_auc']}"
)


# ============================================================
# 7. Display history
# ============================================================

manager.display_history()


print(
    "\n"
    + "=" * 60
)

print(
    "INITIAL REGISTRATION TEST PASSED"
)

print(
    "=" * 60
)