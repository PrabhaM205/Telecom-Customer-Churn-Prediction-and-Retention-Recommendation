import os

from dotenv import load_dotenv
from google import genai
from src.config import get


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# GEMINI API KEY
# ============================================================

API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


if not API_KEY:

    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Add it to the .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# GEMINI FLASH MODEL
# ============================================================

MODEL_NAME = get("llm", "model_name", default="gemini-3.6-flash")


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(prompt):

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if not response.text:

            return (
                "Gemini did not return a response."
            )

        return response.text


    except Exception as error:

        return (
            "Gemini API Error:\n"
            f"{error}"
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("GEMINI FLASH CONNECTION TEST")
    print("=" * 60)


    test_prompt = """
Explain customer churn in one simple sentence.
"""


    response = generate_response(
        test_prompt
    )


    print("\nGemini Response:")

    print(
        response
    )


    print("\n")
    print("=" * 60)
    print("GEMINI FLASH TEST COMPLETED")
    print("=" * 60)