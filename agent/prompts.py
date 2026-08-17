# agent/prompts.py
"""
Centralized prompt templates + deterministic text helpers for the retention
agent workflow. Keeping these separate from logic (offer_generator.py,
nodes.py) means prompt wording can be tuned without touching function code.
"""

import json

# ---------------------------------------------------------------------------
# Diagnosis Agent -- deterministic SHAP feature -> plain language mapping.
# Prefer a deterministic mapping over an LLM call: diagnosis must be
# traceable 1:1 to actual SHAP drivers, never invented.
# ---------------------------------------------------------------------------

FEATURE_EXPLANATIONS = {
    "Contract_Month-to-month": "they are on a month-to-month contract",
    "Tenure Months": "they are a relatively new customer",
    "Online Security_No": "they do not have Online Security",
    "Tech Support_No": "they do not have Tech Support",
    "Internet Service_Fiber optic": "they are on the higher-priced Fiber optic internet plan",
    "Payment Method_Electronic check": "they pay by electronic check (a payment method linked to higher churn)",
    "Monthly Charges": "their monthly bill is relatively high",
    "Total Charges": "of their overall billing history",
    "Paperless Billing_No": "they are not enrolled in paperless billing",
    "Multiple Lines_No": "they only have a single line",
    "Streaming TV_No": "they do not have the Streaming TV add-on",
    "Streaming Movies_No": "they do not have the Streaming Movies add-on",
    "Partner_No": "they do not have a partner on the account",
    "Dependents_No": "they do not have dependents on the account",
    "High Risk Contract": "they match a high-risk month-to-month + electronic-check pattern",
    "CLTV": "of their overall lifetime value profile",
}

_DEFAULT_FEATURE_EXPLANATION = "of the '{feature}' factor identified by the churn model"


def explain_feature(feature: str) -> str:
    """Deterministic feature -> plain-language phrase. Never invents reasons."""
    if feature in FEATURE_EXPLANATIONS:
        return FEATURE_EXPLANATIONS[feature]
    for key, phrase in FEATURE_EXPLANATIONS.items():
        if key in feature or feature in key:
            return phrase
    return _DEFAULT_FEATURE_EXPLANATION.format(feature=feature)


def build_diagnosis(shap_drivers: list, churn_probability: float, top_n: int = 3) -> dict:
    """
    Converts SHAP top risk drivers into a concise, plain-language diagnosis.
    Deterministic -- no LLM call. Every driver in the summary is traceable
    back to an actual entry in shap_drivers (no invented reasons).

    shap_drivers: list of {"feature": ..., "shap_value": ...} (already sorted
    by contribution, as produced by src.explainability.shap.explain_customer).
    """
    if churn_probability >= 0.75:
        risk_phrase = "high churn risk"
    elif churn_probability >= 0.50:
        risk_phrase = "moderate churn risk"
    else:
        risk_phrase = "relatively low churn risk"

    top_drivers = shap_drivers[:top_n]
    phrases = [explain_feature(d["feature"]) for d in top_drivers]

    if phrases:
        if len(phrases) == 1:
            reason_text = phrases[0]
        elif len(phrases) == 2:
            reason_text = f"{phrases[0]} and {phrases[1]}"
        else:
            reason_text = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
        summary = f"This customer is at {risk_phrase} primarily because {reason_text}."
    else:
        summary = f"This customer is at {risk_phrase}, though no dominant SHAP driver was identified."

    return {
        "summary": summary,
        "risk_drivers": top_drivers,
        "churn_probability": round(float(churn_probability), 4),
    }


# ---------------------------------------------------------------------------
# Offer-Strategist Agent -- structured JSON offer generation prompt.
# ---------------------------------------------------------------------------

OFFER_JSON_SCHEMA_INSTRUCTIONS = """
Return ONLY a single JSON object (no markdown fences, no commentary before
or after it) with EXACTLY these fields:

{
  "offer_type": "<short offer category, e.g. 'contract_discount', 'bill_credit', 'service_addon_trial', 'no_offer'>",
  "offer_text": "<one or two sentence description of the offer>",
  "reason": "<why this offer targets the customer's specific diagnosed churn driver>",
  "discount_percent": <integer 0-100, 0 if not a discount offer>,
  "duration_months": <integer, number of months the discount/credit applies, 0 if not applicable>,
  "minimum_term_months": <integer, minimum contract term this offer requires the customer to commit to, 0 if none>,
  "incentive_types": [<list of strings, e.g. ["discount", "bill_credit", "service_addon"]>],
  "policy_basis": "<which retrieved policy passage supports this offer>",
  "customer_message": "<short, professional message that could be sent to the customer>"
}

Rules:
1. Use ONLY the RETRIEVED COMPANY POLICY below as the source of truth for what offers, discounts, and terms exist.
2. Do not invent a discount percentage, duration, or benefit that is not supported by the policy text.
3. The offer MUST directly target the customer's diagnosed churn driver(s) below -- do not give a generic offer.
4. If no policy-supported offer exists, set "offer_type" to "no_offer" and explain why in "reason".
5. Output must be valid JSON and nothing else.
"""

RETENTION_OFFER_PROMPT_TEMPLATE = """
You are a telecom customer retention offer-strategist.

DIAGNOSIS (why this customer is at risk)
=========================================
{diagnosis_summary}

Top SHAP risk drivers:
{shap_drivers_text}

CUSTOMER INFORMATION
====================
Customer Data:
{customer_data}

Churn Probability:
{churn_probability:.2f}

Customer Segment:
{customer_segment}

RETRIEVED COMPANY POLICY
========================
{policy_context}

{schema_instructions}
"""

RETENTION_OFFER_RETRY_PROMPT_TEMPLATE = """
You are a telecom customer retention offer-strategist.

Your PREVIOUS offer for this customer was REJECTED by the policy guardrail.

Previous offer (JSON):
{previous_offer}

Guardrail status: {guardrail_status}
Violated rule(s): {violations}
Guardrail feedback: {guardrail_feedback}

You must produce a NEW, genuinely different, policy-compliant offer that
fixes the violation(s) above. Do not repeat the same offer_type/discount
combination that was rejected.

DIAGNOSIS (why this customer is at risk)
=========================================
{diagnosis_summary}

Top SHAP risk drivers:
{shap_drivers_text}

CUSTOMER INFORMATION
====================
Customer Data:
{customer_data}

Churn Probability:
{churn_probability:.2f}

Customer Segment:
{customer_segment}

RETRIEVED COMPANY POLICY
========================
{policy_context}

{schema_instructions}
"""


def _format_shap_drivers(shap_drivers: list) -> str:
    if not shap_drivers:
        return "(none available)"
    lines = []
    for d in shap_drivers:
        lines.append(f"- {d.get('feature')} (shap_value={d.get('shap_value')})")
    return "\n".join(lines)


def build_offer_prompt(diagnosis, customer_data, churn_probability, customer_segment, policy_context) -> str:
    return RETENTION_OFFER_PROMPT_TEMPLATE.format(
        diagnosis_summary=diagnosis.get("summary", ""),
        shap_drivers_text=_format_shap_drivers(diagnosis.get("risk_drivers", [])),
        customer_data=customer_data,
        churn_probability=churn_probability,
        customer_segment=customer_segment,
        policy_context=policy_context,
        schema_instructions=OFFER_JSON_SCHEMA_INSTRUCTIONS,
    )


def build_retry_prompt(
    diagnosis,
    customer_data,
    churn_probability,
    customer_segment,
    policy_context,
    previous_offer,
    guardrail_result,
) -> str:
    return RETENTION_OFFER_RETRY_PROMPT_TEMPLATE.format(
        previous_offer=json.dumps(previous_offer, default=str),
        guardrail_status=guardrail_result.get("status"),
        violations=", ".join(guardrail_result.get("violations", [])) or "(none listed)",
        guardrail_feedback=guardrail_result.get("feedback", ""),
        diagnosis_summary=diagnosis.get("summary", ""),
        shap_drivers_text=_format_shap_drivers(diagnosis.get("risk_drivers", [])),
        customer_data=customer_data,
        churn_probability=churn_probability,
        customer_segment=customer_segment,
        policy_context=policy_context,
        schema_instructions=OFFER_JSON_SCHEMA_INSTRUCTIONS,
    )


def format_diagnosis_summary(diagnosis: dict) -> str:
    """Short display-friendly line for the diagnosis (used by the UI layer)."""
    if not diagnosis:
        return "No diagnosis available."
    return diagnosis.get("summary", "No diagnosis summary available.")
