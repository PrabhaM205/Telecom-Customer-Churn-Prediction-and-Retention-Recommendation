import pandas as pd

from src.data.loader import load_raw_data

from src.monitor.drift_monitor import (
    DriftMonitor
)

from src.monitor.retraining_manager import (
    retrain_from_incoming_data
)


print("=" * 60)
print("DRIFT + AUTOMATIC RETRAINING TEST")
print("=" * 60)


# ============================================================
# 1. Load reference data
# ============================================================

print("\n1. Loading reference data...")

reference_df = load_raw_data()

print(
    "Reference data shape:",
    reference_df.shape
)


# ============================================================
# 2. Create incoming data
# ============================================================

print("\n2. Creating incoming data...")

incoming_df = reference_df.sample(
    n=1000,
    random_state=42
).copy()

print(
    "Incoming data shape:",
    incoming_df.shape
)


# ============================================================
# 3. Create artificial drift
# ============================================================

print("\n3. Creating artificial drift...")

print(
    "Increasing Monthly Charges by 80% "
    "to simulate production drift."
)


incoming_df["Monthly Charges"] = (
    pd.to_numeric(
        incoming_df["Monthly Charges"],
        errors="coerce"
    )
    * 1.80
)


print(
    "Artificial drift created successfully."
)


# ============================================================
# 4. Features to monitor
# ============================================================

features = [

    "Tenure Months",

    "Monthly Charges",

    "Total Charges"

]

print("\n4. Features monitored:")

for feature in features:

    print(
        f"- {feature}"
    )


# ============================================================
# 5. Create drift monitor
# ============================================================

print(
    "\n5. Creating drift monitor..."
)

monitor = DriftMonitor(
    threshold=0.20
)

print(
    "PSI threshold:",
    monitor.threshold
)


# ============================================================
# 6. Calculate PSI
# ============================================================

print(
    "\n6. Calculating PSI..."
)

results = monitor.check_feature_drift(

    reference_df,

    incoming_df,

    features

)


# ============================================================
# 7. Display drift results
# ============================================================

print("\n")
print("=" * 60)
print("DRIFT RESULTS")
print("=" * 60)

drift_detected = False


for feature, result in results.items():

    print(
        f"\nFeature: {feature}"
    )

    print(
        f"PSI    : {result['psi']}"
    )

    print(
        f"Drift  : {result['drift']}"
    )

    print(
        f"Status : {result['status']}"
    )

    if result["drift"]:

        drift_detected = True


# ============================================================
# 8. Decide whether to retrain
# ============================================================

print("\n")
print("=" * 60)

if drift_detected:

    print(
        "DRIFT DETECTED"
    )

    print("=" * 60)

    print(
        "\nAutomatic model retraining "
        "will now start."
    )


    # --------------------------------------------------------
    # 9. Retrain
    # --------------------------------------------------------

    retraining_result = (
        retrain_from_incoming_data(
            incoming_df
        )
    )


    # --------------------------------------------------------
    # 10. Validate retraining
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print(
        "RETRAINING VALIDATION"
    )
    print("=" * 60)

    if retraining_result["model"] is None:

        raise AssertionError(
            "Retrained model was not created."
        )

    if not (
        0 <= retraining_result["accuracy"] <= 1
    ):

        raise AssertionError(
            "Invalid accuracy value."
        )

    if not (
        0 <= retraining_result["roc_auc"] <= 1
    ):

        raise AssertionError(
            "Invalid ROC-AUC value."
        )

    print(
        "\nRetrained model validation successful."
    )

    print(
        "\nNew model:"
    )

    print(
        retraining_result["model_path"]
    )

    print(
        "\nNew ROC-AUC:",
        f"{retraining_result['roc_auc']:.4f}"
    )

    print("\n")
    print("=" * 60)
    print(
        "DRIFT + AUTOMATIC RETRAINING TEST PASSED"
    )
    print("=" * 60)


else:

    print(
        "NO DRIFT DETECTED"
    )

    print("=" * 60)

    print(
        "\nRetraining was not required."
    )

    print(
        "\nDRIFT TEST PASSED - "
        "NO RETRAINING REQUIRED"
    )