from typing import Any, List, Tuple

import numpy as np

from category.functors import DecisionMonad
from core.config import LAMBDA_IND, LAMBDA_TRANS
from core.state import Action, MotivationalState
from magus.goal_change import DefaultGoalChangeCalculator, GoalChangeCalculator
from magus.profile import DecisionProfile


class MagusDecision(DecisionMonad):
    """
    Generic MAGUS additive decision monad.

    """

    def __init__(
        self,
        profile: DecisionProfile,
        goal_change_calculator: GoalChangeCalculator | None = None,
    ):
        if profile is None:
            raise TypeError("MagusDecision requires an explicit DecisionProfile")
        self.profile = profile
        self.goal_change_calculator = (
            goal_change_calculator
            or DefaultGoalChangeCalculator(self.profile)
        )

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

    def propose_delta_g(
        self,
        state: MotivationalState,
        candidate: Action,
        feedback: Any = None,
    ) -> np.ndarray:
        """
        Compute the calculator-derived Delta G for a selected candidate action.
        """
        return self.goal_change_calculator.delta_g(
            state,
            candidate,
            feedback,
            lambda_ind=LAMBDA_IND,
            lambda_trans=LAMBDA_TRANS,
        )

    def decide(
        self,
        state: MotivationalState,
        candidates: List[Action],
        feedback: Any = None,
    ) -> Tuple[Action, np.ndarray]:
        """
        Scores each candidate action and returns the selected action together
        with its proposed goal update Delta G.
        """
        if not candidates:
            raise ValueError("Must provide at least one candidate action to the decision monad.")

        best_action = None
        best_delta_g = None
        best_score = -float("inf")

        for candidate in candidates:
            total_score = self.score_candidate(state, candidate)
            if total_score > best_score:
                best_score = total_score
                best_action = candidate
                best_delta_g = self.propose_delta_g(state, candidate, feedback)

        return best_action, best_delta_g.copy()
