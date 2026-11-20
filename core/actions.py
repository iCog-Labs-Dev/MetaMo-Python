from dataclasses import dataclass
from typing import Final

import numpy as np

from core.config import (
    G_CURIO,
    G_ETHIC,
    G_HELP,
    G_IND,
    G_NOVEL,
    G_SELF,
    G_SOC,
    G_TRANS,
    NUM_GOALS,
)


@dataclass(frozen=True)
class ActionSpec:
    """Declarative action semantics used to derive MAGUS candidate profiles."""

    planning: str
    execution: str
    mode: str
    dominant_goals: tuple[int, ...]
    supporting_goals: tuple[int, ...] = ()
    opposed_goals: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModeProfile:
    """Reusable projection settings for an action style."""

    dominant_alignment: float
    supporting_alignment: float
    opposed_alignment: float
    overgoal_alignment: float
    base_risk: float


@dataclass(frozen=True)
class ActionProfile:
    """Canonical motivational signature of a catalog action."""

    goal_correlations: tuple[float, ...]
    base_risk: float


ACTION_SPECS: Final[dict[str, ActionSpec]] = {
    "safe_answer": ActionSpec(
        planning="Give a careful, grounded answer using only well-supported claims.",
        execution="Answer cautiously, stick to supported facts, and avoid speculation.",
        mode="stabilizing",
        dominant_goals=(G_HELP,),
        supporting_goals=(G_ETHIC,),
    ),
    "guided_explore": ActionSpec(
        planning="Offer creative but bounded exploration with explicit uncertainty.",
        execution="Explore ideas constructively, but label uncertainty clearly and avoid overclaiming.",
        mode="exploratory",
        dominant_goals=(G_CURIO, G_NOVEL),
        supporting_goals=(G_SELF, G_HELP),
    ),
    "ask_clarifying_question": ActionSpec(
        planning="Ask one focused clarification when the request is ambiguous or underspecified.",
        execution="Ask one short clarifying question instead of giving a full answer.",
        mode="stabilizing",
        dominant_goals=(G_HELP,),
        supporting_goals=(G_ETHIC, G_SOC, G_CURIO),
    ),
    "compare_options": ActionSpec(
        planning="Compare alternatives and explain tradeoffs to support a decision.",
        execution="Present a concise comparison of options and their tradeoffs.",
        mode="balanced",
        dominant_goals=(G_HELP,),
        supporting_goals=(G_CURIO, G_NOVEL, G_SELF, G_SOC),
    ),
    "summarize_source": ActionSpec(
        planning="Summarize the given material faithfully and concisely.",
        execution="Summarize the source faithfully without adding unsupported claims.",
        mode="stabilizing",
        dominant_goals=(G_HELP,),
        supporting_goals=(G_ETHIC,),
    ),
    "decline_risky_request": ActionSpec(
        planning="Refuse a risky or unsafe request and redirect to a safer alternative.",
        execution="Briefly refuse the unsafe request and offer a safe alternative.",
        mode="protective",
        dominant_goals=(G_ETHIC,),
        supporting_goals=(G_HELP,),
        opposed_goals=(G_CURIO, G_NOVEL),
    ),
}

DEFAULT_ACTION_ID: Final[str] = "safe_answer"

ACTION_ID_ALIASES: Final[dict[str, str]] = {
    "risky_exploration": "guided_explore",
    "risky_explore": "guided_explore",
    "explore": "guided_explore",
    "clarify": "ask_clarifying_question",
    "clarifying_question": "ask_clarifying_question",
    "compare": "compare_options",
    "summary": "summarize_source",
    "summarize": "summarize_source",
    "decline": "decline_risky_request",
    "refuse": "decline_risky_request",
}

MODE_PROFILES: Final[dict[str, ModeProfile]] = {
    "protective": ModeProfile(1.00, 0.35, -0.30, 0.50, 0.00),
    "stabilizing": ModeProfile(0.85, 0.45, -0.20, 0.40, 0.04),
    "balanced": ModeProfile(0.80, 0.35, -0.20, 0.30, 0.08),
    "exploratory": ModeProfile(0.90, 0.40, -0.20, 0.55, 0.16),
}


def normalize_action_id(action_id: str) -> str:
    normalized = action_id.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in ACTION_SPECS:
        return normalized
    return ACTION_ID_ALIASES.get(normalized, DEFAULT_ACTION_ID)


def _goal_alignment(spec: ActionSpec) -> tuple[float, ...]:
    mode = MODE_PROFILES[spec.mode]
    correlations = np.zeros(NUM_GOALS, dtype=float)

    for goal_idx in spec.dominant_goals:
        correlations[goal_idx] = max(correlations[goal_idx], mode.dominant_alignment)
    for goal_idx in spec.supporting_goals:
        correlations[goal_idx] = max(correlations[goal_idx], mode.supporting_alignment)
    for goal_idx in spec.opposed_goals:
        correlations[goal_idx] = min(correlations[goal_idx], mode.opposed_alignment)

    safety_alignment = max(correlations[G_HELP], correlations[G_ETHIC], 0.0)
    growth_alignment = max(correlations[G_CURIO], correlations[G_NOVEL], correlations[G_SELF], 0.0)
    correlations[G_IND] = max(correlations[G_IND], safety_alignment * mode.overgoal_alignment)
    correlations[G_TRANS] = max(correlations[G_TRANS], growth_alignment * mode.overgoal_alignment)

    return tuple(float(v) for v in np.clip(correlations, -1.0, 1.0))


def _derive_action_profile(spec: ActionSpec) -> ActionProfile:
    mode = MODE_PROFILES[spec.mode]
    return ActionProfile(
        goal_correlations=_goal_alignment(spec),
        base_risk=mode.base_risk,
    )


ACTION_PROFILES: Final[dict[str, ActionProfile]] = {
    action_id: _derive_action_profile(spec)
    for action_id, spec in ACTION_SPECS.items()
}


def action_profile(action_id: str) -> ActionProfile:
    return ACTION_PROFILES[normalize_action_id(action_id)]


def planning_catalog_text() -> str:
    return "\n".join(
        f'- "{action_id}": {spec.planning}'
        for action_id, spec in ACTION_SPECS.items()
    )


def execution_instruction(action_id: str) -> str:
    normalized = normalize_action_id(action_id)
    return ACTION_SPECS[normalized].execution
