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


# ============================================================
# GEMINI CLIENT
# ============================================================
# Lazily constructed -- do NOT raise at import time. A missing API key (or
# any client construction error) is instead surfaced as a "Gemini API
# Error: ..." string from generate_response(), same as any other runtime
# failure, so callers (genai/offer_generator.py) can catch it and fall back
# to the deterministic offer engine instead of the whole process crashing
# on import. This also lets tests import this module and mock
# generate_response() without needing a real API key.

_client = None


def _get_client():
    global _client
    if _client is None:
        if not API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not found. Add it to the .env file."
            )
        _client = genai.Client(api_key=API_KEY)
    return _client


# ============================================================
# GEMINI FLASH MODEL
# ============================================================

MODEL_NAME = get("llm", "model_name", default="gemini-2.0-flash")


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(prompt):

    try:
        client = _get_client()

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