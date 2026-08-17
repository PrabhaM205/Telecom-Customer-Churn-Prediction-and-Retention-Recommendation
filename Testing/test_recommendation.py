from src.recommendation.priority_engine import calculate_priority
from src.recommendation.retention_rules import evaluate_retention_rules


print("========== RECOMMENDATION TEST ==========")


# ==========================================================
# 1. PRIORITY ENGINE TEST
# ==========================================================

print("\n1. Testing priority engine")

result = calculate_priority(
    churn_probability=0.87,
    revenue_at_risk=24000,
    customer_segment="High Value"
)

print("\nPriority result:")

for key, value in result.items():
    print(f"{key}: {value}")

assert result["churn_score"] == 40
assert result["revenue_score"] == 30
assert result["segment_score"] == 20

assert result["priority_score"] == 90
assert result["priority"] == "CRITICAL"
assert result["risk_level"] == "HIGH"

print("\nPriority engine test successful")


# ==========================================================
# 2. RETENTION RULES - HIGH RISK
# ==========================================================

print("\n2. Testing high-risk retention rules")

customer = {
    "tenure": 36,
    "MonthlyCharges": 95,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check"
}

result = evaluate_retention_rules(
    customer_data=customer,
    churn_probability=0.87,
    revenue_at_risk=24000,
    customer_segment="High Value"
)

print("\nRetention result:")

for key, value in result.items():
    print(f"{key}: {value}")

assert result["eligible"] is True
assert result["risk_level"] == "HIGH"

assert result["churn_probability"] == 0.87
assert result["revenue_at_risk"] == 24000
assert result["customer_segment"] == "High Value"

assert "High predicted churn probability" in result["reasons"]
assert "Customer belongs to a high-value segment" in result["reasons"]
assert "High revenue is at risk" in result["reasons"]

assert "Personalized retention offer" in result["recommended_actions"]
assert "Priority retention treatment" in result["recommended_actions"]
assert "Escalated retention action" in result["recommended_actions"]

print("\nHigh-risk retention rules test successful")


# ==========================================================
# 3. RETENTION RULES - MEDIUM RISK
# ==========================================================

print("\n3. Testing medium-risk retention rules")

customer = {
    "tenure": 12,
    "MonthlyCharges": 60,
    "Contract": "One year",
    "PaymentMethod": "Bank transfer"
}

result = evaluate_retention_rules(
    customer_data=customer,
    churn_probability=0.55,
    revenue_at_risk=7000,
    customer_segment="Standard"
)

print("\nRetention result:")

for key, value in result.items():
    print(f"{key}: {value}")

assert result["eligible"] is True
assert result["risk_level"] == "MEDIUM"

assert "Moderate predicted churn probability" in result["reasons"]

assert (
    "Targeted retention communication"
    in result["recommended_actions"]
)

print("\nMedium-risk retention rules test successful")


# ==========================================================
# 4. RETENTION RULES - LOW RISK
# ==========================================================

print("\n4. Testing low-risk retention rules")

customer = {
    "tenure": 6,
    "MonthlyCharges": 40,
    "Contract": "Two year",
    "PaymentMethod": "Bank transfer"
}

result = evaluate_retention_rules(
    customer_data=customer,
    churn_probability=0.20,
    revenue_at_risk=1000,
    customer_segment="Standard"
)

print("\nRetention result:")

for key, value in result.items():
    print(f"{key}: {value}")

assert result["eligible"] is False
assert result["risk_level"] == "LOW"

assert (
    result["recommended_actions"][0]
    == "No immediate retention offer required"
)

print("\nLow-risk retention rules test successful")


# ==========================================================
# FINAL RESULT
# ==========================================================

print("\n========================================")
print("RECOMMENDATION TEST PASSED")
print("========================================")