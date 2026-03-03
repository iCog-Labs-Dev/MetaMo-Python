import json
import numpy as np
from typing import List
from core.state import Stimulus, Action

def parse_stimulus(llm_json_response: str) -> Stimulus:
    """Parses LLM JSON into a MetaMo Stimulus object."""
    try:
        data = json.loads(llm_json_response)
        return Stimulus(
            novelty=float(data.get("novelty", 0.0)),
            conduciveness=float(data.get("conduciveness", 0.0)),
            risk=float(data.get("risk", 0.0)),
            effort=float(data.get("effort", 0.0))
        )
    except Exception as e:
        print(f"Error parsing stimulus: {e}")
        # Fallback to a neutral stimulus if parsing fails
        return Stimulus(0.1, 0.1, 0.1, 0.1)

def parse_actions(llm_json_response: str) -> List[Action]:
    """Parses LLM JSON into a list of MetaMo Action candidates."""
    try:
        data = json.loads(llm_json_response)
        actions = []
        for item in data.get("candidates", []):
            action = Action(
                id=item["id"],
                goal_correlations=np.array(item["goal_correlations"], dtype=float),
                risk_estimate=float(item["risk_estimate"]),
                delta_g=np.array(item["delta_g"], dtype=float)
            )
            actions.append(action)
        return actions
    except Exception as e:
        print(f"Error parsing actions: {e}")
        # Fallback to a safe default action
        return [Action("default_safe_wait", np.zeros(8), 0.0, np.zeros(8))]