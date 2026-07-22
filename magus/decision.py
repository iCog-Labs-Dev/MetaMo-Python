from typing import List, Tuple

import numpy as np

from category.functors import DecisionMonad
from core.config import LAMBDA_IND, LAMBDA_TRANS
from core.state import Action, MotivationalState
from magus.profile import DecisionProfile


def _default_profile() -> DecisionProfile:
    from applications.research_assistant.decision_profile import (
        RESEARCH_ASSISTANT_DECISION_PROFILE,
    )

    return RESEARCH_ASSISTANT_DECISION_PROFILE


def relevant_modulator(state: MotivationalState, goal_idx: int) -> float:
    """Compatibility helper using the default Research Assistant profile."""
    return _default_profile().relevant_modulator(state, goal_idx)


def overgoal_support(goal_idx: int, g_ind: float, g_trans: float) -> float:
    """Compatibility helper using the default Research Assistant profile."""
    return _default_profile().overgoal_support(goal_idx, g_ind, g_trans)


class MagusDecision(DecisionMonad):
    """
    Generic MAGUS additive decision monad.

    The monad keeps the weighted-additive DS form. Application semantics such
    as goal-modulator relevance, compatibility, and anti-goal penalties are
    supplied by a DecisionProfile.
    """

    def __init__(self, profile: DecisionProfile | None = None):
        self.profile = profile or _default_profile()

    def unit(self, state: MotivationalState) -> MotivationalState:
        """
        The monadic unit (eta). Injects the state into the decision context
        without altering it.
        """
        return state

    def score_candidate(self, state: MotivationalState, candidate: Action) -> float:
        """
        Score a single candidate action under the current motivational state.
        """
        return self.profile.score_candidate(
            state=state,
            candidate=candidate,
            lambda_ind=LAMBDA_IND,
            lambda_trans=LAMBDA_TRANS,
        )

    def decide(self, state: MotivationalState, candidates: List[Action]) -> Tuple[Action, np.ndarray]:
        """
        Scores each candidate action and returns the selected action together
        with its proposed goal update Delta G. The pseudo-bimonad owns the
        finalized state transition after damping and safe-region enforcement.
        """
        if not candidates:
            raise ValueError("Must provide at least one candidate action to the decision monad.")

        best_action = None
        best_score = -float("inf")

        for candidate in candidates:
            total_score = self.score_candidate(state, candidate)
            if total_score > best_score:
                best_score = total_score
                best_action = candidate

        return best_action, best_action.delta_g.copy()
