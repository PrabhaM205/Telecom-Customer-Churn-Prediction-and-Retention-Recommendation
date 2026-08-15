"""
Priority Engine
---------------
Determines customer retention priority using:

- Churn probability
- Revenue at risk
- Customer segment
- Risk level
"""

from typing import Dict


def calculate_priority(
    churn_probability,
    revenue_at_risk=0.0,
    customer_segment="Standard",
    risk_level=None
):
    """
    Calculate retention priority.
    """

    churn_probability = float(churn_probability)
    revenue_at_risk = float(revenue_at_risk)

    segment = str(customer_segment).lower()

    # ---------------------------------------------------------
    # Churn score
    # ---------------------------------------------------------

    if churn_probability >= 0.85:
        churn_score = 40

    elif churn_probability >= 0.70:
        churn_score = 30

    elif churn_probability >= 0.50:
        churn_score = 20

    elif churn_probability >= 0.30:
        churn_score = 10

    else:
        churn_score = 0

    # ---------------------------------------------------------
    # Revenue score
    # ---------------------------------------------------------

    if revenue_at_risk >= 25000:
        revenue_score = 40

    elif revenue_at_risk >= 15000:
        revenue_score = 30

    elif revenue_at_risk >= 10000:
        revenue_score = 20

    elif revenue_at_risk >= 5000:
        revenue_score = 10

    else:
        revenue_score = 0

    # ---------------------------------------------------------
    # Segment score
    # ---------------------------------------------------------

    if segment in [
        "high value",
        "high_value",
        "vip",
        "premium"
    ]:
        segment_score = 20

    elif segment in [
        "medium value",
        "medium_value"
    ]:
        segment_score = 10

    else:
        segment_score = 0

    # ---------------------------------------------------------
    # Total score
    # ---------------------------------------------------------

    priority_score = (
        churn_score
        + revenue_score
        + segment_score
    )

    # Maximum = 100
    priority_score = min(
        priority_score,
        100
    )

    # ---------------------------------------------------------
    # Priority classification
    # ---------------------------------------------------------

    if priority_score >= 75:
        priority = "CRITICAL"

    elif priority_score >= 50:
        priority = "HIGH"

    elif priority_score >= 25:
        priority = "MEDIUM"

    else:
        priority = "LOW"

    # ---------------------------------------------------------
    # Risk level fallback
    # ---------------------------------------------------------

    if risk_level is None:

        if churn_probability >= 0.75:
            risk_level = "HIGH"

        elif churn_probability >= 0.50:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

    return {
        "priority": priority,
        "priority_score": priority_score,
        "risk_level": risk_level,
        "churn_score": churn_score,
        "revenue_score": revenue_score,
        "segment_score": segment_score
    }


if __name__ == "__main__":

    result = calculate_priority(
        churn_probability=0.87,
        revenue_at_risk=24000,
        customer_segment="High Value"
    )

    print("\n" + "=" * 60)
    print("PRIORITY ENGINE TEST")
    print("=" * 60)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 60)