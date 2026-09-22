import json

from applications.research_assistant.prompts import TASK_INTENTS
from applications.research_assistant.stimulus import (
    ResearchAssistantPerception,
    ResearchAssistantSignals,
    ResearchAssistantStimulus,
)


STIMULUS_FIELDS = ("novelty", "conduciveness", "risk", "effort")
SIGNAL_FIELDS = (
    "task_intent",
    "ambiguity",
    "citation_need",
    "comparison_need",
    "summary_need",
    "exploration_need",
    "unsafe_pressure",
    "privacy_pressure",
    "unsupported_claim_pressure",
    "context_loss_pressure",
)


def _load_json_object(llm_json_response: str) -> dict:
    try:
        data = json.loads(llm_json_response)
    except json.JSONDecodeError as error:
        raise ValueError("LLM response is not valid JSON") from error
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object")
    return data


def _require_fields(data: dict, fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in data]
    if missing:
        raise ValueError(f"{label} missing required fields: {missing}")


def parse_stimulus(llm_json_response: str) -> ResearchAssistantStimulus:
    """Parse LLM JSON into a research-assistant stimulus object."""
    data = _load_json_object(llm_json_response)
    _require_fields(data, STIMULUS_FIELDS, "stimulus response")
    return _stimulus_from_mapping(data)


def _stimulus_from_mapping(data: dict) -> ResearchAssistantStimulus:
    return ResearchAssistantStimulus(
        novelty=float(data["novelty"]),
        conduciveness=float(data["conduciveness"]),
        risk=float(data["risk"]),
        effort=float(data["effort"]),
    )


def _signals_from_mapping(data: dict) -> ResearchAssistantSignals:
    _require_fields(data, SIGNAL_FIELDS, "signal response")
    task_intent = str(data["task_intent"]).strip().lower()
    if task_intent not in TASK_INTENTS:
        raise ValueError(f"unknown research assistant task intent: {task_intent}")
    return ResearchAssistantSignals(
        task_intent=task_intent,
        ambiguity=float(data["ambiguity"]),
        citation_need=float(data["citation_need"]),
        comparison_need=float(data["comparison_need"]),
        summary_need=float(data["summary_need"]),
        exploration_need=float(data["exploration_need"]),
        unsafe_pressure=float(data["unsafe_pressure"]),
        privacy_pressure=float(data["privacy_pressure"]),
        unsupported_claim_pressure=float(data["unsupported_claim_pressure"]),
        context_loss_pressure=float(data["context_loss_pressure"]),
    )


def parse_perception(llm_json_response: str) -> ResearchAssistantPerception:
    """Parse LLM JSON into semantic perception for deterministic planning."""
    data = _load_json_object(llm_json_response)
    _require_fields(data, ("stimulus", "signals"), "perception response")
    if not isinstance(data["stimulus"], dict):
        raise ValueError("perception stimulus must be a JSON object")
    if not isinstance(data["signals"], dict):
        raise ValueError("perception signals must be a JSON object")
    _require_fields(data["stimulus"], STIMULUS_FIELDS, "stimulus response")
    return ResearchAssistantPerception(
        stimulus=_stimulus_from_mapping(data["stimulus"]),
        signals=_signals_from_mapping(data["signals"]),
    )
