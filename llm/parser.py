import json

from core.actions import normalize_action_id
from core.state import Stimulus


def _bounded_float(value: object, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a float") from error
    return max(0.0, min(1.0, parsed))


def parse_stimulus(llm_json_response: str) -> Stimulus:
    """Parse LLM JSON into a validated MetaMo Stimulus object."""
    data = json.loads(llm_json_response)
    return Stimulus(
        novelty=_bounded_float(data["novelty"], "novelty"),
        conduciveness=_bounded_float(data["conduciveness"], "conduciveness"),
        risk=_bounded_float(data["risk"], "risk"),
        effort=_bounded_float(data["effort"], "effort"),
    )


def parse_action_risks(llm_json_response: str) -> dict[str, float]:
    """Parse LLM JSON into contextual risk estimates keyed by action id."""
    data = json.loads(llm_json_response)
    risk_overrides = {}
    for item in data.get("candidates", []):
        action_id = normalize_action_id(item["id"])
        risk_overrides[action_id] = _bounded_float(item["risk_estimate"], "risk_estimate")
    return risk_overrides
