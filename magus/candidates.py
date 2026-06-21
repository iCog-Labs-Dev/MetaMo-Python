import math
import re
from collections import Counter
from typing import List, Mapping

import numpy as np

from core.actions import ACTION_PROFILES, ACTION_SPECS, ActionSpec, action_profile
from core.state import Action, Stimulus

RISK_PRIOR = 0.5
RISK_CONTEXT_GAIN = 0.5
MIN_CONTEXT_RELEVANCE = 0.15
TEXT_RELEVANCE_WEIGHT = 0.65
AFFORDANCE_RELEVANCE_WEIGHT = 0.35


def _stem_token(token: str) -> str:
    for suffix in ("ization", "ational", "fulness", "iveness", "tion", "ing", "ed", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _tokenize(text: str) -> List[str]:
    return [_stem_token(token) for token in re.findall(r"[a-z']+", text.lower())]


def _cosine_similarity(left: List[str], right: List[str]) -> float:
    if not left or not right:
        return 0.0
    left_counts = Counter(left)
    right_counts = Counter(right)
    shared = set(left_counts) & set(right_counts)
    numerator = sum(left_counts[token] * right_counts[token] for token in shared)
    left_norm = math.sqrt(sum(count * count for count in left_counts.values()))
    right_norm = math.sqrt(sum(count * count for count in right_counts.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _action_tokens(action_id: str, spec: ActionSpec) -> List[str]:
    action_text = " ".join([
        action_id.replace("_", " "),
        spec.mode,
        spec.planning,
        spec.execution,
    ])
    return _tokenize(action_text)


def _risk_evidence(stimulus_risk: float, risk_override: float | None = None) -> float:
    excess_appraised_risk = max(0.0, stimulus_risk - RISK_PRIOR) / max(1.0 - RISK_PRIOR, 1e-9)
    if risk_override is None:
        return excess_appraised_risk
    return max(excess_appraised_risk, float(np.clip(risk_override, 0.0, 1.0)))


def _mode_affordance(
    spec: ActionSpec,
    stimulus: Stimulus,
    risk_override: float | None = None,
) -> float:
    benign_novelty = stimulus.novelty * (1.0 - stimulus.risk)
    if spec.mode == "protective":
        return _risk_evidence(stimulus.risk, risk_override)
    if spec.mode == "exploratory":
        return benign_novelty
    if spec.mode == "balanced":
        return (stimulus.conduciveness + stimulus.novelty + stimulus.effort) / 3.0
    return (stimulus.conduciveness + stimulus.risk + (1.0 - stimulus.novelty)) / 3.0


def _context_relevance(
    action_id: str,
    stimulus: Stimulus,
    document_tokens: List[str],
    risk_override: float | None = None,
) -> float:
    spec = ACTION_SPECS[action_id]
    text_fit = _cosine_similarity(document_tokens, _action_tokens(action_id, spec))
    affordance_fit = _mode_affordance(spec, stimulus, risk_override)
    if spec.mode == "protective":
        risk_fit = _risk_evidence(stimulus.risk, risk_override)
        return float(np.clip(max(text_fit, affordance_fit, risk_fit), 0.0, 1.0))

    relevance = (TEXT_RELEVANCE_WEIGHT * text_fit) + (
        AFFORDANCE_RELEVANCE_WEIGHT * affordance_fit
    )
    return float(np.clip(relevance, MIN_CONTEXT_RELEVANCE, 1.0))


def _risk_estimate(
    action_id: str,
    stimulus: Stimulus,
    risk_override: float | None,
) -> float:
    profile = action_profile(action_id)
    if risk_override is not None:
        return float(np.clip(max(profile.base_risk, risk_override), 0.0, 1.0))
    if action_id == "decline_risky_request":
        return profile.base_risk
    excess_risk = _risk_evidence(stimulus.risk)
    return float(np.clip(profile.base_risk + RISK_CONTEXT_GAIN * excess_risk, 0.0, 1.0))


def _build_action(
    action_id: str,
    stimulus: Stimulus,
    document_tokens: List[str],
    risk_override: float | None = None,
) -> Action:
    profile = action_profile(action_id)
    relevance = _context_relevance(action_id, stimulus, document_tokens, risk_override)
    return Action(
        id=action_id,
        goal_correlations=np.array(profile.goal_correlations, dtype=float) * relevance,
        risk_estimate=_risk_estimate(action_id, stimulus, risk_override),
    )


def build_candidate_actions(
    document_text: str,
    stimulus: Stimulus,
    risk_overrides: Mapping[str, float] | None = None,
) -> List[Action]:
    """
    Build the full MAGUS action vocabulary with context relevance and risk estimates.

    The candidate factory never selects the action. It only supplies the paper's
    candidate-specific relevance/risk terms so the decision monad can score all actions.
    """
    document_tokens = _tokenize(document_text)
    risk_overrides = risk_overrides or {}
    return [
        _build_action(
            action_id,
            stimulus,
            document_tokens,
            risk_overrides.get(action_id),
        )
        for action_id in ACTION_PROFILES
    ]
