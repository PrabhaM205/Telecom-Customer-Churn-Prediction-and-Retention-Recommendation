# agent/nodes.py
"""
The 4 agent nodes of the retention workflow:

    diagnosis_node        -- Diagnosis Agent
    offer_strategist_node -- Offer-Strategist Agent
    guardrail_node         -- Guardrail Agent
    orchestrator_node      -- Orchestrator Agent (workflow control ONLY --
                               never generates offers)
"""

from agent.state import RetentionAgentState
from agent.prompts import build_diagnosis
from genai.offer_generator import generate_retention_offer
from src.recommendation.retention_rules import evaluate_offer_policy
from src.config import get

MAX_RETRIES = get("agent", "max_retries", default=2)
TOP_N_SHAP_DRIVERS = get("agent", "top_n_shap_drivers", default=5)


# ---------------------------------------------------------------------------
# 1. Diagnosis Agent
# ---------------------------------------------------------------------------

def diagnosis_node(state: RetentionAgentState) -> RetentionAgentState:
    """
    Input: SHAP top risk drivers + churn probability + customer profile.
    Output: a concise, plain-language diagnosis, traceable to the actual
    SHAP drivers (deterministic mapping -- no LLM call, see agent/prompts.py).
    """
    diagnosis = build_diagnosis(
        shap_drivers=state.get("shap_drivers") or [],
        churn_probability=state["churn_probability"],
        top_n=TOP_N_SHAP_DRIVERS,
    )
    state["diagnosis"] = diagnosis
    return state


# ---------------------------------------------------------------------------
# 2. Offer-Strategist Agent
# ---------------------------------------------------------------------------

def offer_strategist_node(state: RetentionAgentState) -> RetentionAgentState:
    """
    Generates a structured JSON retention offer that targets the diagnosed
    churn driver, grounded in RAG-retrieved company policy. On a retry
    (after Guardrail rejection), receives the previous candidate offer and
    the exact guardrail violation/feedback so it produces a genuinely NEW
    compliant offer rather than repeating the rejected one.
    """
    is_retry = state.get("guardrail_result") is not None and state["guardrail_result"].get("retryable")

    result = generate_retention_offer(
        diagnosis=state["diagnosis"],
        customer_data=state["customer_data"],
        churn_probability=state["churn_probability"],
        customer_segment=state["customer_segment"],
        guardrail_result=state.get("guardrail_result") if is_retry else None,
        previous_offer=state.get("candidate_offer") if is_retry else None,
    )
    state["candidate_offer"] = result["offer"]
    return state


# ---------------------------------------------------------------------------
# 3. Guardrail Agent
# ---------------------------------------------------------------------------

def guardrail_node(state: RetentionAgentState) -> RetentionAgentState:
    """
    Real policy enforcement (numeric checks run in Python, not just
    prompted) via src.recommendation.retention_rules.evaluate_offer_policy().
    Produces a structured APPROVED / REJECTED / ESCALATE result.
    """
    customer_data_with_segment = dict(state["customer_data"])
    customer_data_with_segment["customer_segment"] = state["customer_segment"]

    guardrail_result = evaluate_offer_policy(
        offer=state["candidate_offer"],
        customer_data=customer_data_with_segment,
    )
    state["guardrail_result"] = guardrail_result
    state["guardrail_feedback"] = guardrail_result.get("feedback")
    return state


# ---------------------------------------------------------------------------
# 4. Orchestrator Agent -- workflow control ONLY, never generates offers
# ---------------------------------------------------------------------------

def orchestrator_node(state: RetentionAgentState) -> RetentionAgentState:
    """
    APPROVED               -> finish, final_offer = candidate_offer
    ESCALATE                -> stop, escalated = True (never retried)
    REJECTED (retryable)    -> increment retry_count; if under max_retries,
                                loop back to Offer-Strategist, else escalate
                                to human (retries exhausted).

    The graph can never loop forever: retry_count is monotonically
    incremented and compared against max_retries (config.yaml-driven,
    single source of truth) every pass through this node.
    """
    guardrail_result = state["guardrail_result"]
    status = guardrail_result["status"]

    if status == "APPROVED":
        state["final_offer"] = state["candidate_offer"]
        state["escalated"] = False
        return state

    if status == "ESCALATE":
        state["escalated"] = True
        state["final_offer"] = None
        return state

    # REJECTED and (by construction) retryable
    state["final_offer"] = None
    state["retry_count"] += 1
    if state["retry_count"] >= state.get("max_retries", MAX_RETRIES):
        state["escalated"] = True

    return state


def route_after_orchestrator(state: RetentionAgentState) -> str:
    if state["guardrail_result"]["status"] == "APPROVED":
        return "end"
    if state["escalated"]:
        return "end"
    return "retry"  # loop back to Offer-Strategist
