# agent/nodes.py
from src.recommendation.retention_rules import evaluate_retention_rules
from genai.offer_generator import generate_retention_offer
from agent.state import RetentionAgentState
from src.config import get

MAX_RETRIES = get("agent", "max_retries", default=2)


def diagnosis_node(state: RetentionAgentState) -> RetentionAgentState:
    """Confirms WHY this customer is eligible for retention action."""
    eligibility = evaluate_retention_rules(
        customer_data=state["customer_data"],
        churn_probability=state["churn_probability"],
        revenue_at_risk=state["revenue_at_risk"],
        customer_segment=state["customer_segment"],
    )
    state["eligibility"] = eligibility
    return state


def offer_strategist_node(state: RetentionAgentState) -> RetentionAgentState:
    """
    THIS is where your already-written GenAI code runs.
    generate_retention_offer() already does RAG retrieval + Gemini call.
    On a retry (after Guardrail rejection), the previous rejection_reason
    is passed in so the LLM doesn't repeat the same rejected offer.
    """
    result = generate_retention_offer(
        customer_data=state["customer_data"],
        churn_probability=state["churn_probability"],
        customer_segment=state["customer_segment"],
        rejection_reason=state.get("rejection_reason"),
    )
    state["offer_result"] = result
    return state


def guardrail_node(state: RetentionAgentState) -> RetentionAgentState:
    """
    Checks the drafted offer before it can go out.
    Since your offer_generator already forces the LLM to only use
    RAG-retrieved policy text (not invent offers), the guardrail here
    mainly confirms: (1) customer IS actually eligible, and (2) the
    LLM didn't say "no eligible offer found."
    """
    eligible = state["eligibility"]["eligible"]
    recommendation_text = state["offer_result"]["recommendation"].lower()

    no_offer_found = "no eligible retention offer" in recommendation_text

    if not eligible or no_offer_found:
        state["approved"] = False
        state["rejection_reason"] = (
            "Customer not eligible" if not eligible else "No policy-supported offer found"
        )
    else:
        state["approved"] = True
        state["rejection_reason"] = None

    return state


def orchestrator_node(state: RetentionAgentState) -> RetentionAgentState:
    """Tracks retries; decides whether to loop back or escalate."""
    if not state["approved"]:
        state["retry_count"] += 1
        if state["retry_count"] >= MAX_RETRIES:
            state["escalated"] = True
    return state