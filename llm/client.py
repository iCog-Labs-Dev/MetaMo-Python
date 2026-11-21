import time
from typing import Callable, Dict, TypeVar

from dotenv import load_dotenv

from core.state import Stimulus
from llm.parser import parse_action_risks, parse_stimulus
from llm.prompts import get_action_risk_prompt, get_appraisal_prompt

GEMINI_MODEL = "gemini-3-flash-preview"
RETRYABLE_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "HIGH DEMAND")
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.5

T = TypeVar("T")

_client = None
_types = None


def is_retryable(error: Exception) -> bool:
    message = str(error).upper()
    return any(marker in message for marker in RETRYABLE_MARKERS)


def call_with_retry(operation: Callable[[], T]) -> T:
    """Run a Gemini call, retrying with linear backoff on transient service failures."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return operation()
        except Exception as error:
            if attempt == MAX_ATTEMPTS - 1 or not is_retryable(error):
                raise
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))


def _gemini_client():
    global _client, _types
    if _client is None:
        load_dotenv()
        from google import genai
        from google.genai import types

        _client = genai.Client()
        _types = types
    return _client, _types


def query_llm_for_json(prompt: str) -> str:
    """Query the LLM and require JSON output, with bounded retry on transient service failures."""

    def generate() -> str:
        client, types = _gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return response.text

    return call_with_retry(generate)


def get_stimulus_from_text(document_text: str) -> Stimulus:
    """Pipeline: Text -> Prompt -> Gemini -> Parser -> Stimulus"""
    prompt = get_appraisal_prompt(document_text)
    json_response = query_llm_for_json(prompt)
    return parse_stimulus(json_response)


def get_action_risks_from_text(document_text: str, current_mood: Dict[str, float]) -> dict[str, float]:
    """Pipeline: Text + Mood -> Prompt -> Gemini -> Parser -> contextual action risks."""
    prompt = get_action_risk_prompt(document_text, current_mood)
    json_response = query_llm_for_json(prompt)
    return parse_action_risks(json_response)
