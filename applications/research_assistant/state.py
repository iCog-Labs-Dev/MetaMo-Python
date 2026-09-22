from typing import Mapping

import numpy as np

from applications.research_assistant.schema import RESEARCH_ASSISTANT_SCHEMA
from core.state import MotivationalState


DEFAULT_GOAL_VALUES = {
    "individuation": 0.5,
    "transcendence": 0.5,
    "help": 0.8,
    "curiosity": 0.6,
    "ethics": 0.9,
    "novelty": 0.4,
    "self_improvement": 0.3,
    "sociality": 0.2,
    "misinformation": 0.35,
    "unsupported_claim": 0.35,
    "privacy_violation": 0.30,
    "unsafe_assistance": 0.35,
    "context_loss": 0.25,
}


def default_goal_vector() -> np.ndarray:
    vector = np.zeros(RESEARCH_ASSISTANT_SCHEMA.num_goals, dtype=float)
    for goal_name, value in DEFAULT_GOAL_VALUES.items():
        vector[RESEARCH_ASSISTANT_SCHEMA.goal_index(goal_name)] = value
    return vector


def make_state(override: Mapping[str | int, float] | None = None) -> MotivationalState:
    """
    Build the default Research Assistant motivational state.
    """
    goals = default_goal_vector()
    if override:
        for key, value in override.items():
            idx = key if isinstance(key, int) else RESEARCH_ASSISTANT_SCHEMA.goal_index(key)
            goals[idx] = value
    modulators = np.full(RESEARCH_ASSISTANT_SCHEMA.num_modulators, 0.5, dtype=float)
    return MotivationalState(G=goals, M=modulators, schema=RESEARCH_ASSISTANT_SCHEMA)
