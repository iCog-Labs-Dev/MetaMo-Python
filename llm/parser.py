import json

from core.actions import resolve_action_id
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
    """Parse LLM JSON into risk estimates per action, skipping unknown ids and invalid risks."""
    data = json.loads(llm_json_response)
    risk_overrides = {}
    for item in data.get("candidates", []):
        raw_id = item.get("id")
        action_id = resolve_action_id(raw_id) if isinstance(raw_id, str) else None
        if action_id is None:
            continue
        try:
            risk_overrides[action_id] = _bounded_float(item.get("risk_estimate"), "risk_estimate")
        except ValueError:
            continue
    return risk_overrides
