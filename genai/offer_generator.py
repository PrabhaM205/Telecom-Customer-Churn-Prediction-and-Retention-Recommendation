from rag.retriever import PolicyRetriever
from llm_client import generate_response


retriever = PolicyRetriever()


# ---------------------------------------------------------
# Generate Retention Offer
# ---------------------------------------------------------

def generate_retention_offer(
    customer_data,
    churn_probability,
    customer_segment="High Value"
):
    """
    Generate a retention recommendation using:

    1. Customer information
    2. Churn probability
    3. Customer segment
    4. Retrieved company policy
    5. Gemini
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
    # Step 4: Create Gemini prompt
    # -----------------------------------------------------

    prompt = f"""
You are a telecom customer retention assistant.

Your task is to recommend the most appropriate
retention offer for the customer.

IMPORTANT RULES:

1. Use ONLY the company policies provided below.
2. Do not invent offers that are not present in the policy.
3. Do not promise anything that is not supported by policy.
4. Give one primary recommendation.
5. Give a short reason for the recommendation.
6. Mention the relevant policy.
7. Keep the recommendation practical and customer-friendly.

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


OUTPUT FORMAT
=============

Recommended Offer:
<one suitable retention offer>

Why this offer:
<short explanation>

Policy Basis:
<mention the policy that supports the recommendation>

Customer Message:
<a short professional message that can be sent to the customer>
"""

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

    # Example customer
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


    # Generate recommendation
    result = generate_retention_offer(
        customer_data=customer_data,
        churn_probability=churn_probability,
        customer_segment=customer_segment
    )


    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

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

    for i, policy in enumerate(
        result["retrieved_policies"],
        start=1
    ):

        print(f"\nPolicy {i}")
        print("-" * 60)

        print("Page:", policy["page"])

        print(
            "Similarity:",
            round(
                policy["similarity_score"],
                4
            )
        )

        print("\nPolicy Text:")
        print(policy["text"][:800])


    print("\n" + "=" * 60)
    print("GEMINI RETENTION RECOMMENDATION")
    print("=" * 60)

    print(
        result["recommendation"]
    )


    print("\n" + "=" * 60)
    print("RETENTION OFFER GENERATION COMPLETED")
    print("=" * 60)