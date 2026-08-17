# agent/state.py
"""
LangGraph state for the 4-agent retention workflow:

    Diagnosis Agent -> Offer-Strategist Agent -> Guardrail Agent -> Orchestrator Agent

Kept as a plain TypedDict (no nested custom classes) so the whole state is
trivially JSON-serializable for logging/debugging and for surfacing in the
Streamlit UI.

IMPORTANT: `diagnosis` (WHY the customer is at risk, derived from SHAP) and
`eligibility` are different concepts and are not conflated here. This state
does not carry a business "eligibility" gate at all -- the Diagnosis Agent's
job is explanation, not gatekeeping. Retention decisioning happens in the
Guardrail Agent, which validates the drafted offer against policy.
"""

from typing import TypedDict, Optional, Dict, Any, List


class RetentionAgentState(TypedDict):
    # ---- Inputs (populated before the graph runs) ----
    customer_data: Dict[str, Any]
    churn_probability: float
    revenue_at_risk: float
    customer_segment: str

    # ---- SHAP explainability (populated before the graph runs) ----
    shap_drivers: List[Dict[str, Any]]

    # ---- Diagnosis Agent fills this ----
    diagnosis: Optional[Dict[str, Any]]

    # ---- Offer-Strategist Agent fills this ----
    candidate_offer: Optional[Dict[str, Any]]

    # ---- Guardrail Agent fills this ----
    guardrail_result: Optional[Dict[str, Any]]
    guardrail_feedback: Optional[str]

    # ---- Orchestrator Agent tracks this ----
    retry_count: int
    max_retries: int
    escalated: bool

    # ---- Final output ----
    final_offer: Optional[Dict[str, Any]]
