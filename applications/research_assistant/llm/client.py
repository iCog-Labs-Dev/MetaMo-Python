import time
from typing import Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai import types

from applications.research_assistant.calibration import calibrate_perception
from applications.research_assistant.llm.parser import parse_perception
from applications.research_assistant.planning import build_candidates
from applications.research_assistant.prompts import (
    get_perception_prompt,
)
from applications.research_assistant.stimulus import (
    ResearchAssistantPerception,
    ResearchAssistantStimulus,
)
from core.state import Action


load_dotenv()
client = genai.Client()

RETRYABLE_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "HIGH DEMAND")


def _is_retryable(error: Exception) -> bool:
    message = str(error).upper()
    return any(marker in message for marker in RETRYABLE_MARKERS)


def query_llm_for_json(prompt: str) -> str:
    """Query the LLM and require JSON output, with bounded retry."""
    last_error = None
    for attempt in range(3):
        try:
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


def get_perception_from_text(document_text: str) -> ResearchAssistantPerception:
    """Pipeline: text -> prompt -> Gemini -> parser -> semantic perception."""
    prompt = get_perception_prompt(document_text)
    json_response = query_llm_for_json(prompt)
    return parse_perception(json_response)


def get_calibrated_perception_from_text(document_text: str) -> ResearchAssistantPerception:
    """Pipeline: text -> prompt -> Gemini -> parser -> calibrated perception."""
    return calibrate_perception(get_perception_from_text(document_text)).perception


def get_stimulus_from_text(document_text: str) -> ResearchAssistantStimulus:
    """Pipeline: text -> perception -> app stimulus."""
    return get_calibrated_perception_from_text(document_text).stimulus


def get_candidates_from_text(document_text: str, current_mood: Dict[str, float]) -> List[Action]:
    """Pipeline: text -> perception -> deterministic Research Assistant candidates."""
    perception = get_calibrated_perception_from_text(document_text)
    return build_candidates(perception, current_mood)
