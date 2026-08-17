import os
import json

from src.models.model_manager import ModelManager


print("=" * 60)
print("MODEL ACCURACY COMPARISON TEST")
print("=" * 60)


# ============================================================
# 1. Create Model Manager
# ============================================================

manager = ModelManager()

print("\n1. Model Manager loaded successfully")


# ============================================================
# 2. Get current production model
# ============================================================

current_version = manager.get_current_version()

if current_version is None:

    raise AssertionError(
        "No production model is registered."
    )


print(
    f"\n2. Current production version: "
    f"{current_version}"
)


# ============================================================
# 3. Get current production metrics
# ============================================================

current_model = manager.get_current_model()

if current_model is None:

    raise AssertionError(
        "Current production model information not found."
    )


old_metrics = {

    "accuracy":
        current_model["accuracy"],

    "precision":
        current_model["precision"],

    "recall":
        current_model["recall"],

    "f1_score":
        current_model["f1_score"],

    "roc_auc":
        current_model["roc_auc"]
}


print("\n3. OLD PRODUCTION MODEL")
print("-" * 40)

print(
    f"Version   : {current_version}"
)

print(
    f"Accuracy  : "
    f"{old_metrics['accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{old_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{old_metrics['recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{old_metrics['f1_score']:.4f}"
)

print(
    f"ROC-AUC   : "
    f"{old_metrics['roc_auc']:.4f}"
)


# ============================================================
# 4. Load newly retrained model metrics
# ============================================================

metrics_path = os.path.join(
    manager.artifacts_dir,
    "xgboost_metrics.json"
)


if not os.path.exists(metrics_path):

    raise FileNotFoundError(
        f"Metrics file not found:\n{metrics_path}"
    )


with open(
    metrics_path,
    "r"
) as file:

    new_metrics = json.load(file)


print("\n4. NEW RETRAINED MODEL")
print("-" * 40)

print(
    f"Accuracy  : "
    f"{new_metrics['accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{new_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{new_metrics['recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{new_metrics['f1_score']:.4f}"
)

print(
    f"ROC-AUC   : "
    f"{new_metrics['roc_auc']:.4f}"
)


# ============================================================
# 5. Compare models
# ============================================================

decision = manager.compare_models(
    old_metrics,
    new_metrics
)


# ============================================================
# 6. Validate decision
# ============================================================

if decision not in [
    "PROMOTE",
    "REJECT"
]:

    raise AssertionError(
        f"Invalid model decision: {decision}"
    )


print("\n5. MODEL DECISION VALIDATED")

print(
    f"Decision: {decision}"
)


# ============================================================
# 7. Display comparison summary
# ============================================================

accuracy_difference = (
    float(new_metrics["accuracy"])
    -
    float(old_metrics["accuracy"])
)

roc_auc_difference = (
    float(new_metrics["roc_auc"])
    -
    float(old_metrics["roc_auc"])
)


print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)

print(
    f"\nOld Accuracy : "
    f"{old_metrics['accuracy']:.4f}"
)

print(
    f"New Accuracy : "
    f"{new_metrics['accuracy']:.4f}"
)

print(
    f"Accuracy Difference : "
    f"{accuracy_difference:+.4f}"
)


print(
    f"\nOld ROC-AUC : "
    f"{old_metrics['roc_auc']:.4f}"
)

print(
    f"New ROC-AUC : "
    f"{new_metrics['roc_auc']:.4f}"
)

print(
    f"ROC-AUC Difference : "
    f"{roc_auc_difference:+.4f}"
)


# ============================================================
# 8. Final result
# ============================================================

print("\n" + "=" * 60)

print(
    "MODEL COMPARISON TEST PASSED"
)

print("=" * 60)