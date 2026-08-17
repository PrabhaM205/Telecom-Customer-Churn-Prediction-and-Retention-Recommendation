import pandas as pd

from src.data.loader import load_raw_data
from src.monitor.drift_monitor import DriftMonitor


print("=" * 60)
print("DRIFT MONITOR TEST")
print("=" * 60)


# ============================================================
# 1. LOAD REFERENCE DATA
# ============================================================

print("\n1. Loading reference data...")

reference_df = load_raw_data()

print(
    "Reference data shape:",
    reference_df.shape
)


# ============================================================
# 2. CREATE INCOMING DATA
# ============================================================

print("\n2. Creating incoming data...")

incoming_df = (
    reference_df
    .sample(
        n=1000,
        random_state=42
    )
    .copy()
)

print(
    "Incoming data shape:",
    incoming_df.shape
)


# ============================================================
# 3. CREATE ARTIFICIAL DRIFT
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
    * 1.8
)


print(
    "Artificial drift created successfully."
)


# ============================================================
# 4. FEATURES TO MONITOR
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
# 5. CREATE DRIFT MONITOR
# ============================================================

print("\n5. Creating drift monitor...")

monitor = DriftMonitor(
    threshold=0.20
)

print(
    "PSI threshold:",
    monitor.threshold
)


# ============================================================
# 6. CALCULATE DRIFT
# ============================================================

print("\n6. Calculating PSI...")


results = monitor.check_feature_drift(
    reference_df,
    incoming_df,
    features
)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n========== DRIFT RESULTS ==========")


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


# ============================================================
# 8. OVERALL DRIFT
# ============================================================

print("\n==========================================")

drift_detected = monitor.has_drift(
    reference_df,
    incoming_df,
    features
)


if drift_detected:

    print(
        "DRIFT DETECTED"
    )

else:

    print(
        "NO SIGNIFICANT DRIFT DETECTED"
    )


# ============================================================
# 9. VALIDATION
# ============================================================

print("\n==========================================")

if not drift_detected:

    raise AssertionError(
        "Expected drift was not detected."
    )


print(
    "Drift detection validation successful"
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n==========================================")
print("DRIFT MONITOR TEST PASSED")
print("==========================================")