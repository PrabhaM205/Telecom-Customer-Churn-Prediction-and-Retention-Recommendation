# tests/test_agent.py
"""
Tests for the 4-agent retention LangGraph workflow (agent/state.py,
agent/nodes.py, agent/graph.py) and the Offer-Strategist's structured-JSON
parsing (genai/offer_generator.py).

The LLM call (genai.llm_client.generate_response) and the RAG retriever
(genai.rag.retriever.PolicyRetriever) are always mocked -- these tests have
no live Gemini API or FAISS-index dependency.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.prompts import build_diagnosis
from agent.nodes import diagnosis_node, guardrail_node, orchestrator_node
from genai.offer_generator import parse_offer_response, generate_retention_offer


# ---------------------------------------------------------------------------
# Diagnosis Agent
# ---------------------------------------------------------------------------

def test_diagnosis_uses_shap_drivers():
    shap_drivers = [
        {"feature": "Contract_Month-to-month", "shap_value": 0.9},
        {"feature": "Tech Support_No", "shap_value": 0.6},
        {"feature": "Payment Method_Electronic check", "shap_value": 0.4},
    ]
    diagnosis = build_diagnosis(shap_drivers, churn_probability=0.82, top_n=3)

    assert diagnosis["churn_probability"] == 0.82
    assert diagnosis["risk_drivers"] == shap_drivers
    # Every driver mentioned must trace back to an actual SHAP feature name
    assert "month-to-month" in diagnosis["summary"].lower()
    assert "tech support" in diagnosis["summary"].lower()
    assert "high churn risk" in diagnosis["summary"].lower()


def test_diagnosis_node_populates_state():
    state = {
        "shap_drivers": [{"feature": "Contract_Month-to-month", "shap_value": 0.9}],
        "churn_probability": 0.6,
    }
    result = diagnosis_node(state)
    assert result["diagnosis"]["churn_probability"] == 0.6
    assert result["diagnosis"]["risk_drivers"][0]["feature"] == "Contract_Month-to-month"


def test_diagnosis_no_drivers_does_not_invent_reasons():
    diagnosis = build_diagnosis([], churn_probability=0.4, top_n=3)
    assert diagnosis["risk_drivers"] == []
    assert "no dominant" in diagnosis["summary"].lower()


# ---------------------------------------------------------------------------
# Offer-Strategist: structured JSON parsing / validation
# ---------------------------------------------------------------------------

VALID_OFFER = {
    "offer_type": "bill_credit",
    "offer_text": "5% bill credit for 3 months.",
    "reason": "Targets high monthly charges.",
    "discount_percent": 5,
    "duration_months": 3,
    "minimum_term_months": 0,
    "incentive_types": ["bill_credit"],
    "policy_basis": "Section 2.1",
    "customer_message": "Thank you for being a valued customer.",
}


def test_parse_offer_response_valid_json():
    offer = parse_offer_response(json.dumps(VALID_OFFER))
    assert offer["offer_type"] == "bill_credit"
    assert offer["discount_percent"] == 5


def test_parse_offer_response_strips_markdown_fences():
    fenced = "```json\n" + json.dumps(VALID_OFFER) + "\n```"
    offer = parse_offer_response(fenced)
    assert offer["offer_type"] == "bill_credit"


def test_parse_offer_response_rejects_malformed_json():
    import pytest
    with pytest.raises(ValueError):
        parse_offer_response("This is not JSON at all, just prose from the LLM.")


def test_parse_offer_response_rejects_missing_fields():
    import pytest
    incomplete = {"offer_type": "bill_credit"}
    with pytest.raises(ValueError):
        parse_offer_response(json.dumps(incomplete))


def test_parse_offer_response_rejects_empty_string():
    import pytest
    with pytest.raises(ValueError):
        parse_offer_response("")


# ---------------------------------------------------------------------------
# Offer-Strategist: end-to-end generation (LLM + RAG mocked)
# ---------------------------------------------------------------------------

DIAGNOSIS = {
    "summary": "This customer is at high churn risk primarily because they are on a "
                "month-to-month contract and do not have Tech Support.",
    "risk_drivers": [
        {"feature": "Contract_Month-to-month", "shap_value": 0.9},
        {"feature": "Tech Support_No", "shap_value": 0.6},
    ],
    "churn_probability": 0.82,
}


def _mock_retriever():
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.search.return_value = [
        {"page": 2, "text": "Retention discounts up to 5% for 0-6 month tenure.", "similarity_score": 0.8}
    ]
    return mock_retriever_instance


@patch("genai.offer_generator._get_retriever")
@patch("genai.offer_generator.generate_response")
def test_valid_llm_offer_is_used_directly(mock_generate_response, mock_get_retriever):
    mock_get_retriever.return_value = _mock_retriever()
    mock_generate_response.return_value = json.dumps(VALID_OFFER)

    result = generate_retention_offer(
        diagnosis=DIAGNOSIS,
        customer_data={"Tenure Months": 2},
        churn_probability=0.82,
        customer_segment="Standard",
    )

    assert result["fallback_used"] is False
    assert result["offer"]["offer_type"] == "bill_credit"


@patch("genai.offer_generator._get_retriever")
@patch("genai.offer_generator.generate_response")
def test_malformed_llm_json_triggers_deterministic_fallback(mock_generate_response, mock_get_retriever):
    mock_get_retriever.return_value = _mock_retriever()
    mock_generate_response.return_value = "Sorry, here is your offer: <not valid json>"

    result = generate_retention_offer(
        diagnosis=DIAGNOSIS,
        customer_data={"Tenure Months": 2},
        churn_probability=0.82,
        customer_segment="Standard",
    )

    assert result["fallback_used"] is True
    # Fallback offer must still be well-formed
    assert "offer_type" in result["offer"]
    assert "discount_percent" in result["offer"]


@patch("genai.offer_generator._get_retriever")
@patch("genai.offer_generator.generate_response")
def test_llm_failure_triggers_deterministic_fallback(mock_generate_response, mock_get_retriever):
    mock_get_retriever.return_value = _mock_retriever()
    mock_generate_response.return_value = "Gemini API Error:\nConnection timeout"

    result = generate_retention_offer(
        diagnosis=DIAGNOSIS,
        customer_data={"Tenure Months": 2},
        churn_probability=0.82,
        customer_segment="Standard",
    )

    assert result["fallback_used"] is True
    assert result["offer"]["offer_type"] in ("contract_discount", "service_addon_trial", "bill_credit")


@patch("genai.offer_generator._get_retriever")
def test_rag_retriever_exception_does_not_crash(mock_get_retriever):
    mock_get_retriever.side_effect = RuntimeError("FAISS index not found")

    with patch("genai.offer_generator.generate_response", return_value=json.dumps(VALID_OFFER)):
        result = generate_retention_offer(
            diagnosis=DIAGNOSIS,
            customer_data={"Tenure Months": 2},
            churn_probability=0.82,
            customer_segment="Standard",
        )
    # Should not raise -- policy_context degrades to an error string, LLM
    # (mocked here) still runs, but the overall call must not crash.
    assert "offer" in result


# ---------------------------------------------------------------------------
# Guardrail Agent node
# ---------------------------------------------------------------------------

def test_guardrail_node_approves_valid_offer():
    state = {
        "candidate_offer": VALID_OFFER,
        "customer_data": {"Tenure Months": 2},
        "customer_segment": "Standard",
    }
    result = guardrail_node(state)
    assert result["guardrail_result"]["status"] == "APPROVED"
    assert result["guardrail_feedback"]


def test_guardrail_node_rejects_offer_above_cap():
    offer = dict(VALID_OFFER, discount_percent=50)
    state = {
        "candidate_offer": offer,
        "customer_data": {"Tenure Months": 2},
        "customer_segment": "Standard",
    }
    result = guardrail_node(state)
    assert result["guardrail_result"]["status"] == "REJECTED"
    assert "TENURE_DISCOUNT_CAP" in result["guardrail_result"]["violations"]


# ---------------------------------------------------------------------------
# Orchestrator Agent: retry / escalation logic
# ---------------------------------------------------------------------------

def test_orchestrator_approved_finishes_with_final_offer():
    state = {
        "candidate_offer": VALID_OFFER,
        "guardrail_result": {"status": "APPROVED", "approved": True, "retryable": False, "violations": [], "feedback": "ok"},
        "retry_count": 0,
        "max_retries": 2,
        "escalated": False,
    }
    result = orchestrator_node(state)
    assert result["final_offer"] == VALID_OFFER
    assert result["escalated"] is False


def test_orchestrator_retries_on_rejected_retryable_offer():
    state = {
        "candidate_offer": VALID_OFFER,
        "guardrail_result": {"status": "REJECTED", "approved": False, "retryable": True,
                              "violations": ["TENURE_DISCOUNT_CAP"], "feedback": "too high"},
        "retry_count": 0,
        "max_retries": 2,
        "escalated": False,
    }
    result = orchestrator_node(state)
    assert result["retry_count"] == 1
    assert result["escalated"] is False
    assert result["final_offer"] is None


def test_orchestrator_escalates_after_max_retries_exhausted():
    state = {
        "candidate_offer": VALID_OFFER,
        "guardrail_result": {"status": "REJECTED", "approved": False, "retryable": True,
                              "violations": ["TENURE_DISCOUNT_CAP"], "feedback": "too high"},
        "retry_count": 1,  # already at max_retries - 1
        "max_retries": 2,
        "escalated": False,
    }
    result = orchestrator_node(state)
    assert result["retry_count"] == 2
    assert result["escalated"] is True
    assert result["final_offer"] is None


def test_orchestrator_escalate_status_never_retries():
    state = {
        "candidate_offer": VALID_OFFER,
        "guardrail_result": {"status": "ESCALATE", "approved": False, "retryable": False,
                              "violations": ["PERMANENT_DISCOUNT"], "feedback": "needs human"},
        "retry_count": 0,
        "max_retries": 2,
        "escalated": False,
    }
    result = orchestrator_node(state)
    assert result["escalated"] is True
    assert result["final_offer"] is None
    # retry_count untouched -- escalation short-circuits before incrementing
    assert result["retry_count"] == 0


# ---------------------------------------------------------------------------
# Full graph: retry produces a genuinely new offer, not a repeat
# ---------------------------------------------------------------------------

@patch("genai.offer_generator._get_retriever")
@patch("genai.offer_generator.generate_response")
def test_retry_produces_new_offer_not_a_repeat(mock_generate_response, mock_get_retriever):
    """
    Simulates: first LLM offer breaches the tenure discount cap (REJECTED,
    retryable) -> Offer-Strategist is called again with guardrail feedback
    -> second LLM offer is compliant. Verifies the two offers differ and the
    workflow does not repeat the rejected discount.
    """
    mock_get_retriever.return_value = _mock_retriever()

    rejected_offer = dict(VALID_OFFER, discount_percent=50)  # breaches 0-6mo cap of 5%
    compliant_offer = dict(VALID_OFFER, discount_percent=5)

    mock_generate_response.side_effect = [
        json.dumps(rejected_offer),
        json.dumps(compliant_offer),
    ]

    from src.recommendation.retention_rules import evaluate_offer_policy

    customer_data = {"Tenure Months": 2, "customer_segment": "Standard"}

    # Attempt 1
    result1 = generate_retention_offer(
        diagnosis=DIAGNOSIS, customer_data=customer_data,
        churn_probability=0.82, customer_segment="Standard",
    )
    guardrail1 = evaluate_offer_policy(result1["offer"], customer_data)
    assert guardrail1["status"] == "REJECTED"
    assert guardrail1["retryable"] is True

    # Attempt 2 (retry, with guardrail feedback)
    result2 = generate_retention_offer(
        diagnosis=DIAGNOSIS, customer_data=customer_data,
        churn_probability=0.82, customer_segment="Standard",
        guardrail_result=guardrail1, previous_offer=result1["offer"],
    )
    guardrail2 = evaluate_offer_policy(result2["offer"], customer_data)
    assert guardrail2["status"] == "APPROVED"
    assert result1["offer"]["discount_percent"] != result2["offer"]["discount_percent"]
