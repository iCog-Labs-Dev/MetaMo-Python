from collections.abc import Mapping
from typing import Any

import numpy as np

from applications.research_assistant.actions import ACTION_SPECS
from applications.research_assistant.schema import RESEARCH_ASSISTANT_SCHEMA
from applications.research_assistant.stimulus import (
    ResearchAssistantPerception,
    ResearchAssistantSignals,
    clamp_unit,
)
from core.state import Action


INTENT_ACTIONS = {
    "answer": "safe_answer",
    "summarize": "summarize_source",
    "compare": "compare_options",
    "clarify": "ask_clarifying_question",
    "explore": "guided_explore",
    "decline": "decline_risky_request",
}


def _mood_value(current_mood: Mapping[str, Any] | None, name: str) -> float:
    if not current_mood:
        return 0.5
    return clamp_unit(float(current_mood.get(name, 0.5)))


def _goal_vector(values: Mapping[str, float]) -> np.ndarray:
    vector = np.zeros(RESEARCH_ASSISTANT_SCHEMA.num_goals, dtype=float)
    for goal_name, value in values.items():
        vector[RESEARCH_ASSISTANT_SCHEMA.goal_index(goal_name)] = float(value)
    return np.clip(vector, -1.0, 1.0)


def _delta_vector(values: Mapping[str, float]) -> np.ndarray:
    vector = np.zeros(RESEARCH_ASSISTANT_SCHEMA.num_goals, dtype=float)
    for goal_name, value in values.items():
        vector[RESEARCH_ASSISTANT_SCHEMA.goal_index(goal_name)] = float(value)
    return np.clip(vector, -0.1, 0.1)


def _intent_boost(signals: ResearchAssistantSignals, action_id: str) -> float:
    return 0.20 if INTENT_ACTIONS.get(signals.task_intent) == action_id else 0.0


def _metadata(
    action_id: str,
    signals: ResearchAssistantSignals,
    risk: float,
) -> dict[str, Any]:
    return {
        "risk": clamp_unit(risk),
        "generated_by": "deterministic_research_assistant_profile",
        "task_intent": signals.task_intent,
        "action_policy": action_id,
    }


def build_candidates(
    perception: ResearchAssistantPerception,
    current_mood: Mapping[str, Any] | None = None,
) -> list[Action]:
    """
    Build Research Assistant candidates deterministically from semantic signals.
    """
    signals = perception.signals
    caution = _mood_value(current_mood, "caution")
    arousal = _mood_value(current_mood, "arousal")

    unsafe = signals.unsafe_pressure
    privacy = signals.privacy_pressure
    unsupported = signals.unsupported_claim_pressure
    context_loss = signals.context_loss_pressure
    ambiguity = signals.ambiguity
    citation = signals.citation_need
    summary = signals.summary_need
    comparison = signals.comparison_need
    exploration = signals.exploration_need
    safety_pressure = max(unsafe, privacy)

    specs: dict[str, tuple[dict[str, float], float, dict[str, float]]] = {
        "safe_answer": (
            {
                "help": 0.75 + 0.10 * citation,
                "curiosity": 0.15 + 0.10 * exploration,
                "novelty": 0.05,
                "self_improvement": 0.10,
                "ethics": 0.70 + 0.15 * caution,
                "sociality": 0.35,
                "misinformation": 0.10 + 0.30 * unsupported,
                "unsupported_claim": 0.12 + 0.35 * unsupported,
                "privacy_violation": 0.05 + 0.30 * privacy,
                "unsafe_assistance": 0.05 + 0.30 * unsafe,
                "context_loss": 0.08 + 0.20 * context_loss,
            },
            0.06 + 0.35 * unsupported + 0.25 * privacy + 0.25 * unsafe,
            {"help": 0.02, "ethics": 0.01},
        ),
        "guided_explore": (
            {
                "help": 0.35,
                "curiosity": 0.75 + 0.20 * exploration + 0.10 * arousal,
                "novelty": 0.75 + 0.15 * exploration,
                "self_improvement": 0.25,
                "ethics": 0.30,
                "sociality": 0.35,
                "misinformation": 0.20 + 0.35 * unsupported,
                "unsupported_claim": 0.22 + 0.40 * unsupported,
                "privacy_violation": 0.10 + 0.35 * privacy,
                "unsafe_assistance": 0.15 + 0.55 * unsafe,
                "context_loss": 0.10 + 0.20 * context_loss,
            },
            0.14 + 0.40 * unsupported + 0.25 * privacy + 0.45 * unsafe,
            {"curiosity": 0.025, "novelty": 0.025},
        ),
        "ask_clarifying_question": (
            {
                "help": 0.45 + 0.20 * ambiguity - 0.20 * safety_pressure,
                "curiosity": 0.30,
                "novelty": 0.10,
                "self_improvement": 0.15,
                "ethics": 0.70 + 0.15 * ambiguity,
                "sociality": 0.70 + 0.15 * ambiguity - 0.25 * safety_pressure,
                "misinformation": -0.20,
                "unsupported_claim": -0.35,
                "privacy_violation": -0.20,
                "unsafe_assistance": -0.10,
                "context_loss": -0.45,
            },
            0.03 + 0.10 * privacy,
            {"help": 0.01, "sociality": 0.02, "context_loss": -0.02},
        ),
        "compare_options": (
            {
                "help": 0.65 + 0.20 * comparison,
                "curiosity": 0.40 + 0.10 * comparison,
                "novelty": 0.20,
                "self_improvement": 0.20,
                "ethics": 0.55,
                "sociality": 0.45,
                "misinformation": 0.10 + 0.20 * unsupported,
                "unsupported_claim": 0.12 + 0.25 * unsupported,
                "privacy_violation": 0.08 + 0.20 * privacy,
                "unsafe_assistance": 0.08 + 0.20 * unsafe,
                "context_loss": 0.08 + 0.15 * context_loss,
            },
            0.07 + 0.25 * unsupported + 0.20 * privacy + 0.20 * unsafe,
            {"help": 0.015, "self_improvement": 0.01},
        ),
        "summarize_source": (
            {
                "help": 0.70 + 0.20 * summary,
                "curiosity": 0.15,
                "novelty": 0.10,
                "self_improvement": 0.15,
                "ethics": 0.70 + 0.10 * citation,
                "sociality": 0.35,
                "misinformation": 0.06 + 0.20 * unsupported,
                "unsupported_claim": 0.08 + 0.25 * unsupported,
                "privacy_violation": 0.08 + 0.25 * privacy,
                "unsafe_assistance": 0.05 + 0.20 * unsafe,
                "context_loss": -0.35 + 0.15 * context_loss,
            },
            0.05 + 0.20 * unsupported + 0.20 * privacy + 0.15 * unsafe,
            {"help": 0.02, "ethics": 0.015, "context_loss": -0.02},
        ),
        "decline_risky_request": (
            {
                "help": 0.20 + 0.25 * safety_pressure,
                "curiosity": -0.10,
                "novelty": -0.10,
                "self_improvement": 0.05,
                "ethics": 0.85 + 0.15 * safety_pressure,
                "sociality": 0.25 + 0.20 * safety_pressure,
                "misinformation": -0.25,
                "unsupported_claim": -0.25,
                "privacy_violation": -0.70,
                "unsafe_assistance": -0.85,
                "context_loss": 0.05,
            },
            0.02,
            {"ethics": 0.03, "unsafe_assistance": -0.03, "privacy_violation": -0.02},
        ),
    }

    candidates = []
    for action_id in ACTION_SPECS:
        correlations, risk, delta = specs[action_id]
        correlations = dict(correlations)
        correlations["help"] = correlations.get("help", 0.0) + _intent_boost(signals, action_id)
        candidates.append(
            Action(
                id=action_id,
                goal_correlations=_goal_vector(correlations),
                delta_g=_delta_vector(delta),
                schema=RESEARCH_ASSISTANT_SCHEMA,
                metadata=_metadata(action_id, signals, risk),
            )
        )

    return candidates
