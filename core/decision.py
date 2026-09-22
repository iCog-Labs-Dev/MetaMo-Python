from dataclasses import dataclass
from typing import Iterator, Mapping

import numpy as np

from core.state import Action


@dataclass(frozen=True)
class CandidateScoreBreakdown:
    """Explainable score components for one candidate action."""

    action_id: str
    primary_goal_contributions: Mapping[str, float]
    anti_goal_penalties: Mapping[str, float]
    individuation_adjustment: float
    transcendence_adjustment: float
    expected_goal_change: float
    total_score: float


@dataclass(frozen=True)
class DecisionResult:
    """A selected action, its already-computed update, and all score evidence.

    Iteration preserves the former ``action, delta_g = decide(...)`` contract.
    """

    action: Action
    proposed_delta_g: np.ndarray
    chosen_score: CandidateScoreBreakdown
    candidate_scores: tuple[CandidateScoreBreakdown, ...]

    def __iter__(self) -> Iterator[object]:
        yield self.action
        yield self.proposed_delta_g

