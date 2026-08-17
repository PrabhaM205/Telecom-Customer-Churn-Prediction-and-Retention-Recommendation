# agent/graph.py
from langgraph.graph import StateGraph, END
from agent.state import RetentionAgentState
from agent.nodes import (
    diagnosis_node,
    offer_strategist_node,
    guardrail_node,
    orchestrator_node,
    route_after_orchestrator,
    MAX_RETRIES,
)


def build_retention_agent():
    """
    4-agent retention workflow:

        Diagnosis -> Offer-Strategist -> Guardrail -> Orchestrator
                          ^                                |
                          |________________ retry __________|
                                    (REJECTED, under max_retries)

    Orchestrator ends the graph on APPROVED or ESCALATE. retry_count is
    monotonically incremented in orchestrator_node and bounded by
    max_retries, so the retry loop always terminates.
    """
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
        "churn_probability": 0.82,
        "revenue_at_risk": 300.00,
        "customer_segment": "Standard",
        "shap_drivers": [
            {"feature": "Contract_Month-to-month", "shap_value": 0.9},
            {"feature": "Tech Support_No", "shap_value": 0.6},
            {"feature": "Payment Method_Electronic check", "shap_value": 0.4},
        ],
        "diagnosis": None,
        "candidate_offer": None,
        "guardrail_result": None,
        "guardrail_feedback": None,
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
        "escalated": False,
        "final_offer": None,
    }

    final_state = app.invoke(initial_state)

    print("=" * 60)
    print("AGENT WORKFLOW RESULT")
    print("=" * 60)
    print(f"Guardrail Status : {final_state['guardrail_result']['status']}")
    print(f"Escalated        : {final_state['escalated']}")
    print(f"Retry Count      : {final_state['retry_count']}")
    if final_state["final_offer"]:
        print("\nFinal Offer:\n", final_state["final_offer"])
    elif final_state["escalated"]:
        print(f"\nEscalated: {final_state['guardrail_feedback']}")
