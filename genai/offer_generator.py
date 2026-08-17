import json
import re

from genai.rag.retriever import PolicyRetriever
from genai.llm_client import generate_response
from agent.prompts import build_offer_prompt, build_retry_prompt
from src.recommendation.offer_engine import generate_fallback_offer

_retriever = None


def _get_retriever():
    """Lazy singleton -- avoids loading the FAISS index / embedder at import
    time (e.g. during `import genai.offer_generator` for unit tests that
    mock the LLM/retriever)."""
    global _retriever
    if _retriever is None:
        _retriever = PolicyRetriever()
    return _retriever


# ---------------------------------------------------------
# Structured-offer JSON parsing / validation
# ---------------------------------------------------------

REQUIRED_OFFER_FIELDS = [
    "offer_type", "offer_text", "reason", "discount_percent",
    "duration_months", "minimum_term_months", "incentive_types",
    "policy_basis", "customer_message",
]


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_offer_response(raw_text: str) -> dict:
    """
    Robustly parses an LLM's structured-offer response into a dict.
    Strips markdown code fences, extracts the first {...} JSON object if
    there's surrounding commentary, and validates required fields.

    Raises ValueError on any malformed output -- callers MUST catch this
    and fall back to a deterministic offer. Arbitrary LLM text is never
    silently treated as a valid offer.
    """
    if not raw_text or not isinstance(raw_text, str):
        raise ValueError("Empty or non-string LLM response.")

    cleaned = _strip_code_fences(raw_text)

    # If there's stray text around the JSON object, extract the outermost {...}
    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

    try:
        offer = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM response was not valid JSON: {e}")

    if not isinstance(offer, dict):
        raise ValueError("Parsed LLM response was not a JSON object.")

    missing = [f for f in REQUIRED_OFFER_FIELDS if f not in offer]
    if missing:
        raise ValueError(f"LLM offer JSON missing required field(s): {missing}")

    # Type coercion / defensive normalization
    try:
        offer["discount_percent"] = int(float(offer.get("discount_percent") or 0))
        offer["duration_months"] = int(float(offer.get("duration_months") or 0))
        offer["minimum_term_months"] = int(float(offer.get("minimum_term_months") or 0))
    except (TypeError, ValueError) as e:
        raise ValueError(f"LLM offer JSON had non-numeric numeric field: {e}")

    if not isinstance(offer.get("incentive_types"), list):
        offer["incentive_types"] = [offer["incentive_types"]] if offer.get("incentive_types") else []

    return offer


# ---------------------------------------------------------
# Policy retrieval
# ---------------------------------------------------------

def _retrieve_policy_context(query: str, top_k: int = 3):
    try:
        retriever = _get_retriever()
        policy_results = retriever.search(query, top_k=top_k)
    except Exception as e:
        # RAG index missing/unavailable -- degrade gracefully, offer
        # generation falls back to the deterministic engine below.
        return [], f"(policy retrieval unavailable: {e})"

    policy_context = ""
    for i, result in enumerate(policy_results, start=1):
        policy_context += (
            f"\nPolicy {i}\n"
            f"Page: {result.get('page')}\n"
            f"Similarity Score: {result.get('similarity_score', 0):.4f}\n\n"
            f"{result.get('text', '')}\n"
            f"{'-' * 50}\n"
        )
    return policy_results, policy_context


# ---------------------------------------------------------
# Generate Retention Offer (Offer-Strategist Agent)
# ---------------------------------------------------------

def generate_retention_offer(
    diagnosis: dict,
    customer_data: dict,
    churn_probability: float,
    customer_segment: str = "Standard",
    guardrail_result: dict = None,
    previous_offer: dict = None,
):
    """
    The Offer-Strategist Agent. Produces a structured JSON retention offer
    that specifically targets the customer's diagnosed churn driver(s),
    grounded in retrieved company policy (RAG).

    diagnosis: output of agent.prompts.build_diagnosis() -- the Diagnosis
        Agent's summary + top SHAP risk drivers. Required so the offer is
        traceable to *why* this customer is at risk, not generic.
    guardrail_result / previous_offer: passed in on a RETRY (after the
        Guardrail Agent rejected the previous offer), so the LLM sees the
        exact violated rule and feedback and must produce a genuinely new,
        compliant offer -- never repeat the rejected one.

    Never raises for LLM/RAG failure -- always returns a usable structured
    offer (falling back to a deterministic, policy-tier-based offer when
    the LLM/RAG path fails or returns malformed output).
    """
    policy_query = f"""
    Retention offers and policies suitable for a {customer_segment} telecom
    customer with churn probability {churn_probability:.2f}.
    Diagnosed churn driver(s): {diagnosis.get('summary', '')}
    Customer details: {customer_data}
    """

    policy_results, policy_context = _retrieve_policy_context(policy_query, top_k=3)

    is_retry = guardrail_result is not None and previous_offer is not None
    if is_retry:
        prompt = build_retry_prompt(
            diagnosis=diagnosis,
            customer_data=customer_data,
            churn_probability=churn_probability,
            customer_segment=customer_segment,
            policy_context=policy_context,
            previous_offer=previous_offer,
            guardrail_result=guardrail_result,
        )
    else:
        prompt = build_offer_prompt(
            diagnosis=diagnosis,
            customer_data=customer_data,
            churn_probability=churn_probability,
            customer_segment=customer_segment,
            policy_context=policy_context,
        )

    fallback_used = False
    fallback_reason = None
    offer = None

    try:
        raw_response = generate_response(prompt)
        if raw_response.startswith("Gemini API Error") or raw_response == "Gemini did not return a response.":
            raise ValueError(raw_response)
        offer = parse_offer_response(raw_response)
    except Exception as e:
        fallback_used = True
        fallback_reason = str(e)
        offer = generate_fallback_offer(
            customer_data=customer_data,
            diagnosis=diagnosis,
            churn_probability=churn_probability,
            customer_segment=customer_segment,
            previous_offer=previous_offer,
        )

    return {
        "offer": offer,
        "retrieved_policies": policy_results,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RETENTION OFFER GENERATION TEST")
    print("=" * 60)

    customer_data = {
        "Tenure Months": 4,
        "Monthly Charges": 95.90,
        "Contract": "Month-to-month",
        "Internet Service": "Fiber optic",
        "Payment Method": "Electronic check",
    }

    diagnosis = {
        "summary": "This customer is at high churn risk primarily because they are on a "
                    "month-to-month contract and do not have Tech Support.",
        "risk_drivers": [
            {"feature": "Contract_Month-to-month", "shap_value": 0.9},
            {"feature": "Tech Support_No", "shap_value": 0.6},
        ],
        "churn_probability": 0.82,
    }

    result = generate_retention_offer(
        diagnosis=diagnosis,
        customer_data=customer_data,
        churn_probability=0.82,
        customer_segment="High Value",
    )

    print(json.dumps(result["offer"], indent=2))
    print("\nFallback used:", result["fallback_used"], result["fallback_reason"])
