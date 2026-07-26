from typing import Dict, List

from applications.research_assistant.calibration import calibrate_perception
from applications.research_assistant.llm.parser import parse_perception
from applications.research_assistant.llm.providers import generate_text
from applications.research_assistant.planning import build_candidates
from applications.research_assistant.prompts import (
    get_perception_prompt,
)
from applications.research_assistant.stimulus import (
    ResearchAssistantPerception,
    ResearchAssistantStimulus,
)
from core.state import Action


def query_llm_for_json(prompt: str) -> str:
    """Query the configured LLM and require JSON output."""
    return generate_text(prompt, json_mode=True, temperature=0.2)


def get_perception_from_text(document_text: str) -> ResearchAssistantPerception:
    """Pipeline: text -> prompt -> LLM -> parser -> semantic perception."""
    prompt = get_perception_prompt(document_text)
    json_response = query_llm_for_json(prompt)
    return parse_perception(json_response)


def get_calibrated_perception_from_text(document_text: str) -> ResearchAssistantPerception:
    """Pipeline: text -> prompt -> LLM -> parser -> calibrated perception."""
    return calibrate_perception(get_perception_from_text(document_text)).perception


def get_stimulus_from_text(document_text: str) -> ResearchAssistantStimulus:
    """Pipeline: text -> perception -> app stimulus."""
    return get_calibrated_perception_from_text(document_text).stimulus


def get_candidates_from_text(document_text: str, current_mood: Dict[str, float]) -> List[Action]:
    """Pipeline: text -> perception -> deterministic Research Assistant candidates."""
    perception = get_calibrated_perception_from_text(document_text)
    return build_candidates(perception, current_mood)
