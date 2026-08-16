from genai.rag.retriever import PolicyRetriever
from genai.llm_client import generate_response
from agent.prompts import build_offer_prompt, build_retry_prompt


retriever = PolicyRetriever()


# ---------------------------------------------------------
# Generate Retention Offer
# ---------------------------------------------------------

def generate_retention_offer(
    customer_data,
    churn_probability,
    customer_segment="High Value",
    rejection_reason=None
):
    """
    Generate a retention recommendation using:

    1. Customer information
    2. Churn probability
    3. Customer segment
    4. Retrieved company policy
    5. Gemini

    rejection_reason: passed in by agent/nodes.py on a RETRY attempt,
    after the Guardrail Agent rejected the previous offer. None on the
    first attempt.
    """

    # -----------------------------------------------------
    # Step 1: Create policy search query
    # -----------------------------------------------------

    policy_query = f"""
    Retention offers and policies suitable for a
    {customer_segment} telecom customer with
    churn probability {churn_probability:.2f}.
    Customer details:
    {customer_data}
    """

    # -----------------------------------------------------
    # Step 2: Retrieve relevant policy from FAISS
    # -----------------------------------------------------

    policy_results = retriever.search(
        policy_query,
        top_k=3
    )

    # -----------------------------------------------------
    # Step 3: Build policy context
    # -----------------------------------------------------

    policy_context = ""

    for i, result in enumerate(policy_results, start=1):

        policy_context += f"""
Policy {i}
Page: {result['page']}
Similarity Score: {result['similarity_score']:.4f}

{result['text']}
--------------------------------------------------
"""

    # -----------------------------------------------------
    # Step 4: Build the prompt — uses agent/prompts.py now,
    # picks the retry template if this is a second attempt
    # -----------------------------------------------------

    if rejection_reason:
        prompt = build_retry_prompt(
            customer_data=customer_data,
            churn_probability=churn_probability,
            customer_segment=customer_segment,
            policy_context=policy_context,
            rejection_reason=rejection_reason,
        )
    else:
        prompt = build_offer_prompt(
            customer_data=customer_data,
            churn_probability=churn_probability,
            customer_segment=customer_segment,
            policy_context=policy_context,
        )

    # -----------------------------------------------------
    # Step 5: Send prompt to Gemini
    # -----------------------------------------------------

    response = generate_response(prompt)

    return {
        "customer_data": customer_data,
        "churn_probability": churn_probability,
        "customer_segment": customer_segment,
        "retrieved_policies": policy_results,
        "recommendation": response
    }


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("RETENTION OFFER GENERATION TEST")
    print("=" * 60)

    customer_data = {
        "tenure_months": 48,
        "monthly_charges": 1200,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
        "support_calls": 4
    }

    churn_probability = 0.82
    customer_segment = "High Value"

    result = generate_retention_offer(
        customer_data=customer_data,
        churn_probability=churn_probability,
        customer_segment=customer_segment
    )

    print("\nCUSTOMER")
    print("-" * 60)
    print(customer_data)

    print("\nChurn Probability:")
    print(churn_probability)

    print("\nCustomer Segment:")
    print(customer_segment)

    print("\n" + "=" * 60)
    print("RETRIEVED POLICIES")
    print("=" * 60)

    for i, policy in enumerate(result["retrieved_policies"], start=1):
        print(f"\nPolicy {i}")
        print("-" * 60)
        print("Page:", policy["page"])
        print("Similarity:", round(policy["similarity_score"], 4))
        print("\nPolicy Text:")
        print(policy["text"][:800])

    print("\n" + "=" * 60)
    print("GEMINI RETENTION RECOMMENDATION")
    print("=" * 60)
    print(result["recommendation"])

    print("\n" + "=" * 60)
    print("RETENTION OFFER GENERATION COMPLETED")
    print("=" * 60)