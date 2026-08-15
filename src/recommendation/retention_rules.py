"""
Retention Rules
---------------
Defines business rules for customer retention.

Input:
    customer_data
    churn_probability
    revenue_at_risk
    customer_segment

Output:
    retention eligibility and recommended action
"""


def evaluate_retention_rules(
    customer_data,
    churn_probability,
    revenue_at_risk=0.0,
    customer_segment="Standard"
):
    """
    Evaluate whether a customer should receive a retention action.
    """

    # ---------------------------------------------------------
    # Basic values
    # ---------------------------------------------------------

    churn_probability = float(churn_probability)
    revenue_at_risk = float(revenue_at_risk)

    tenure = float(
        customer_data.get(
            "tenure",
            customer_data.get("Tenure", 0)
        )
    )

    monthly_charges = float(
        customer_data.get(
            "monthly_charges",
            customer_data.get("MonthlyCharges", 0)
        )
    )

    contract = str(
        customer_data.get(
            "contract",
            customer_data.get("Contract", "")
        )
    ).lower()

    payment_method = str(
        customer_data.get(
            "payment_method",
            customer_data.get("PaymentMethod", "")
        )
    ).lower()

    # ---------------------------------------------------------
    # Risk classification
    # ---------------------------------------------------------

    if churn_probability >= 0.75:
        risk_level = "HIGH"

    elif churn_probability >= 0.50:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    # ---------------------------------------------------------
    # Retention eligibility
    # ---------------------------------------------------------

    eligible = False
    reasons = []
    recommended_actions = []

    # High churn customers
    if churn_probability >= 0.75:
        eligible = True

        reasons.append(
            "High predicted churn probability"
        )

        recommended_actions.append(
            "Personalized retention offer"
        )

    # Medium churn customers
    elif churn_probability >= 0.50:
        eligible = True

        reasons.append(
            "Moderate predicted churn probability"
        )

        recommended_actions.append(
            "Targeted retention communication"
        )

    # ---------------------------------------------------------
    # High-value customer
    # ---------------------------------------------------------

    if customer_segment.lower() in [
        "high value",
        "high_value",
        "premium",
        "vip"
    ]:
        eligible = True

        reasons.append(
            "Customer belongs to a high-value segment"
        )

        recommended_actions.append(
            "Priority retention treatment"
        )

    # ---------------------------------------------------------
    # Revenue at risk
    # ---------------------------------------------------------

    if revenue_at_risk >= 10000:

        eligible = True

        reasons.append(
            "High revenue is at risk"
        )

        recommended_actions.append(
            "Escalated retention action"
        )

    # ---------------------------------------------------------
    # Contract-related risk
    # ---------------------------------------------------------

    if "month-to-month" in contract:

        reasons.append(
            "Customer is on a month-to-month contract"
        )

        recommended_actions.append(
            "Offer long-term contract incentive"
        )

    # ---------------------------------------------------------
    # High monthly charges
    # ---------------------------------------------------------

    if monthly_charges >= 80:

        reasons.append(
            "Customer has relatively high monthly charges"
        )

        recommended_actions.append(
            "Consider eligible pricing or plan discount"
        )

    # ---------------------------------------------------------
    # Electronic check payment
    # ---------------------------------------------------------

    if "electronic check" in payment_method:

        reasons.append(
            "Customer uses electronic check payment"
        )

    # ---------------------------------------------------------
    # Tenure
    # ---------------------------------------------------------

    if tenure >= 24:

        reasons.append(
            "Customer has significant tenure"
        )

        recommended_actions.append(
            "Consider loyalty-based retention treatment"
        )

    # ---------------------------------------------------------
    # Remove duplicate actions
    # ---------------------------------------------------------

    recommended_actions = list(
        dict.fromkeys(recommended_actions)
    )

    reasons = list(
        dict.fromkeys(reasons)
    )

    # ---------------------------------------------------------
    # Default action
    # ---------------------------------------------------------

    if not recommended_actions:

        recommended_actions.append(
            "No immediate retention offer required"
        )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    return {
        "eligible": eligible,
        "risk_level": risk_level,
        "reasons": reasons,
        "recommended_actions": recommended_actions,
        "customer_segment": customer_segment,
        "churn_probability": churn_probability,
        "revenue_at_risk": revenue_at_risk
    }


if __name__ == "__main__":

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

    print("\n" + "=" * 60)
    print("RETENTION RULE TEST")
    print("=" * 60)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 60)