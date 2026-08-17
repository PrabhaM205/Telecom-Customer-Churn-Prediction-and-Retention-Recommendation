from src.risk.riske import (
    assign_risk_tier,
    compute_revenue_at_risk
)

import numpy as np
import pandas as pd


print("========== RISK ENGINE TEST ==========")


# 1. Risk tier classification
print("\n1. Testing risk tier classification")

test_cases = [
    (0.20, "Low"),
    (0.50, "Medium"),
    (0.70, "High"),
    (0.90, "Critical"),
]

for probability, expected in test_cases:

    result = assign_risk_tier(probability)

    print(
        f"Probability: {probability:.0%} -> Risk: {result}"
    )

    assert result == expected, (
        f"Expected {expected}, got {result}"
    )

print("Risk tier classification successful")


# 2. Boundary testing
print("\n2. Testing probability boundaries")

assert assign_risk_tier(0.0) == "Low"
assert assign_risk_tier(1.0) == "Critical"

print("Probability boundary testing successful")


# 3. Revenue-at-risk testing
print("\n3. Testing revenue-at-risk calculation")

probabilities = np.array([
    0.50,
    0.80
])

monthly_charges = pd.Series([
    100.0,
    200.0
])

revenue = compute_revenue_at_risk(
    probabilities,
    monthly_charges
)

print(revenue)


assert revenue.loc[0, "monthly_revenue_at_risk"] == 50.0
assert revenue.loc[1, "monthly_revenue_at_risk"] == 160.0

assert revenue.loc[0, "annual_revenue_at_risk"] == 600.0
assert revenue.loc[1, "annual_revenue_at_risk"] == 1920.0

print("Revenue-at-risk calculation successful")


print("\n========================================")
print("RISK ENGINE TEST PASSED")
print("========================================")