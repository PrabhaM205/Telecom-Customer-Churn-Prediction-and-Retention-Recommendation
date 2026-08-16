# agent/state.py
from typing import TypedDict, Optional, Dict, Any

class RetentionAgentState(TypedDict):
    customer_data: Dict[str, Any]
    churn_probability: float
    revenue_at_risk: float
    customer_segment: str

    # Diagnosis Agent fills this
    eligibility: Optional[Dict[str, Any]]

    # Offer-Strategist Agent fills this
    offer_result: Optional[Dict[str, Any]]

    # Guardrail Agent fills this
    approved: Optional[bool]
    rejection_reason: Optional[str]

    # Orchestrator tracks this
    retry_count: int
    escalated: bool