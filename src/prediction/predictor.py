# src/prediction/predictor.py
"""
Single entry point for a full customer prediction: risk score + SHAP
explanation + retention recommendation. Every other part of the app
(Streamlit pages, API, tests) should call predict_customer() here instead
of calling risk/SHAP/agent code directly.

Deliberately does NOT reimplement model loading or preprocessing -- reuses
src/risk/riske.py and src/explainability/shap.py, which already do this
correctly (clean -> engineer -> transform -> predict), so there's only one
place this logic can drift out of sync with training.
"""

import pandas as pd

from src.risk.riske import load_artifacts as load_risk_artifacts, score_customers
from src.explainability.shap import (
    get_explainer,
    prepare_features,
    compute_shap_values,
    explain_customer,
    suggest_retention_strategy,
)
from agent.graph import build_retention_agent
from agent.state import RetentionAgentState


# --------------------------------------------------
# Cache artifacts across calls -- loading the model/preprocessor/explainer
# is the expensive part. Do it once per process, not once per customer.
# --------------------------------------------------

_model = None
_explainer = None
_agent_app = None


def _get_cached_artifacts():
    global _model, _explainer, _agent_app
    if _model is None:
        _model, _ = load_risk_artifacts()
        _explainer = get_explainer(_model)
    if _agent_app is None:
        _agent_app = build_retention_agent()
    return _model, _explainer, _agent_app


def _infer_segment(annual_revenue_at_risk: float) -> str:
    """Simple placeholder segmentation until a dedicated model exists."""
    if annual_revenue_at_risk >= 15000:
        return "High Value"
    elif annual_revenue_at_risk >= 5000:
        return "Medium Value"
    return "Standard"


# --------------------------------------------------
# Main entry point -- ONE customer, full pipeline
# --------------------------------------------------

def predict_customer(customer_row: dict, run_agent: bool = True) -> dict:
    """
    customer_row: dict of raw column -> value, same schema as a row from
    src.data.loader.load_raw_data().

    run_agent=False skips the GenAI/agent step -- use this for cheap batch
    scoring of a whole file; only run the agent for the top-priority slice
    (see the cost-optimization discussion) by calling this again per
    customer with run_agent=True.
    """
    model, explainer, agent_app = _get_cached_artifacts()
    df_raw = pd.DataFrame([customer_row])

    # ---- 1. Risk score: model + risk tier + revenue at risk ----
    # riske.score_customers() already runs clean -> engineer -> transform -> predict
    risk_result = score_customers(df_raw, model=model).iloc[0].to_dict()

    # ---- 2. SHAP explanation: why ----
    X_transformed, feature_names, _ = prepare_features(df_raw)
    shap_explanation = compute_shap_values(explainer, X_transformed)
    explanation = explain_customer(
        shap_explanation, 0, feature_names, customer_id=risk_result["CustomerID"]
    )
    explanation["retention_strategy"] = suggest_retention_strategy(explanation["top_risk_drivers"])

    result = {
        "customer_id": risk_result["CustomerID"],
        "churn_probability": risk_result["churn_probability"],
        "risk_tier": risk_result["risk_tier"],
        "monthly_revenue_at_risk": risk_result["monthly_revenue_at_risk"],
        "annual_revenue_at_risk": risk_result["annual_revenue_at_risk"],
        "priority_score": risk_result["priority_score"],
        "top_risk_drivers": explanation["top_risk_drivers"],
        "top_retention_drivers": explanation["top_retention_drivers"],
        "retention_strategy": explanation["retention_strategy"],
        "offer": None,
        "agent_approved": None,
        "agent_escalated": None,
    }

    if not run_agent:
        return result

    # ---- 3. Agent workflow: Diagnosis -> Offer-Strategist -> Guardrail -> Orchestrator ----
    initial_state: RetentionAgentState = {
        "customer_data": customer_row,
        "churn_probability": risk_result["churn_probability"],
        "revenue_at_risk": risk_result["annual_revenue_at_risk"],
        "customer_segment": _infer_segment(risk_result["annual_revenue_at_risk"]),
        "eligibility": None,
        "offer_result": None,
        "approved": None,
        "rejection_reason": None,
        "retry_count": 0,
        "escalated": False,
    }

    final_state = agent_app.invoke(initial_state)

    result["agent_approved"] = final_state["approved"]
    result["agent_escalated"] = final_state["escalated"]
    if final_state["approved"]:
        result["offer"] = final_state["offer_result"]["recommendation"]
    elif final_state["escalated"]:
        result["escalation_reason"] = final_state["rejection_reason"]

    return result


# --------------------------------------------------
# Batch entry point -- cheap path (no agent) for scoring a whole file
# --------------------------------------------------

def predict_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    model, _, _ = _get_cached_artifacts()
    return score_customers(df_raw, model=model)


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PREDICTOR FACADE TEST -- single customer, full pipeline")
    print("=" * 60)

    test_customer = {
        "CustomerID": "1771-OADNZ",
        "Tenure Months": 2,
        "Monthly Charges": 95.90,
        "Total Charges": 191.80,
        "Contract": "Month-to-month",
        "Payment Method": "Electronic check",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Paperless Billing": "Yes",
        "Multiple Lines": "No",
        "Phone Service": "Yes",
        "Gender": "Female",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "CLTV": 2151,
    }

    result = predict_customer(test_customer)

    print(f"\nCustomer ID       : {result['customer_id']}")
    print(f"Churn Probability : {result['churn_probability']:.2%}")
    print(f"Risk Tier         : {result['risk_tier']}")
    print(f"Revenue at Risk   : ${result['annual_revenue_at_risk']:,.2f}")
    print(f"Priority Score    : {result['priority_score']}")

    print("\nTop Risk Drivers:")
    for d in result["top_risk_drivers"][:3]:
        print(f"  - {d['feature']} (impact: {d['shap_value']})")

    print(f"\nAgent Approved    : {result['agent_approved']}")
    print(f"Agent Escalated   : {result['agent_escalated']}")
    if result["offer"]:
        print(f"\nFinal Offer:\n{result['offer']}")
    elif result["agent_escalated"]:
        print(f"\nEscalated: {result.get('escalation_reason')}")

    print("\n" + "=" * 60)