import time
from typing import Dict

from dotenv import load_dotenv

from core.state import Stimulus
from llm.parser import parse_action_risks, parse_stimulus
from llm.prompts import get_action_risk_prompt, get_appraisal_prompt

RETRYABLE_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "HIGH DEMAND")
_client = None
_types = None


def _is_retryable(error: Exception) -> bool:
    message = str(error).upper()
    return any(marker in message for marker in RETRYABLE_MARKERS)


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
    last_error = None
    for attempt in range(3):
        try:
            client, types = _gemini_client()
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            return response.text
        except Exception as error:
            last_error = error
            if attempt == 2 or not _is_retryable(error):
                raise
            time.sleep(1.5 * (attempt + 1))

    raise last_error


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
