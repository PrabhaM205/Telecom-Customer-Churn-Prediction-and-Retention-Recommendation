# agent/graph.py
from langgraph.graph import StateGraph, END
from agent.state import RetentionAgentState
from agent.nodes import (
    diagnosis_node,
    offer_strategist_node,
    guardrail_node,
    orchestrator_node,
    MAX_RETRIES,
)


def route_after_orchestrator(state: RetentionAgentState) -> str:
    if state["approved"]:
        return "end"
    if state["escalated"]:
        return "end"
    return "retry"  # loop back to Offer-Strategist


def build_retention_agent():
    workflow = StateGraph(RetentionAgentState)

    workflow.add_node("diagnosis", diagnosis_node)
    workflow.add_node("offer_strategist", offer_strategist_node)
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("orchestrator", orchestrator_node)

    workflow.set_entry_point("diagnosis")
    workflow.add_edge("diagnosis", "offer_strategist")
    workflow.add_edge("offer_strategist", "guardrail")
    workflow.add_edge("guardrail", "orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"retry": "offer_strategist", "end": END},
    )

    return workflow.compile()


if __name__ == "__main__":
    app = build_retention_agent()

    initial_state: RetentionAgentState = {
        "customer_data": {
            "CustomerID": "1771-OADNZ",
            "Tenure Months": 2,
            "Monthly Charges": 95.90,
            "Contract": "Month-to-month",
            "Payment Method": "Electronic check",
        },
        "churn_probability": 0.15,
        "revenue_at_risk": 300.00,
        "customer_segment": "Standard",
        "eligibility": None,
        "offer_result": None,
        "approved": None,
        "rejection_reason": None,
        "retry_count": 0,
        "escalated": False,
    }

    final_state = app.invoke(initial_state)

    print("=" * 60)
    print("AGENT WORKFLOW RESULT")
    print("=" * 60)
    print(f"Approved   : {final_state['approved']}")
    print(f"Escalated  : {final_state['escalated']}")
    if final_state["approved"]:
        print("\nFinal Offer:\n", final_state["offer_result"]["recommendation"])
    elif final_state["escalated"]:
        print(f"\nEscalated after {final_state['retry_count']} attempts: {final_state['rejection_reason']}")