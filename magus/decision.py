from typing import Any, List

import numpy as np

from category.functors import DecisionMonad
from core.config import LAMBDA_IND, LAMBDA_TRANS
from core.state import Action, MotivationalState
from magus.goal_change import DefaultGoalChangeCalculator, GoalChangeCalculator
from magus.profile import DecisionProfile
from core.decision import CandidateScoreBreakdown, DecisionResult


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

    def score_candidate(
        self,
        state: MotivationalState,
        candidate: Action,
        feedback: Any = None,
    ) -> float:
        """
        Score a single candidate action under the current motivational state.
        """
        delta = self.propose_delta_g(state, candidate, feedback)
        return self.profile.score_breakdown(
            state,
            candidate,
            delta,
            LAMBDA_IND,
            LAMBDA_TRANS,
        ).total_score

    def evaluate_candidate(
        self,
        state: MotivationalState,
        candidate: Action,
        feedback: Any = None,
    ) -> tuple[np.ndarray, CandidateScoreBreakdown]:
        """Compute a candidate update and score it once."""
        delta = self.propose_delta_g(state, candidate, feedback)
        breakdown = self.profile.score_breakdown(
            state,
            candidate,
            delta,
            LAMBDA_IND,
            LAMBDA_TRANS,
        )
        return delta, breakdown

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
    ) -> DecisionResult:
        """
        Scores each candidate action and returns the selected action together
        with its proposed goal update Delta G.
        """
        if not candidates:
            raise ValueError("Must provide at least one candidate action to the decision monad.")

        best_action = None
        best_delta_g = None
        best_breakdown = None
        candidate_scores = []

        for candidate in candidates:
            delta_g, breakdown = self.evaluate_candidate(
                state,
                candidate,
                feedback,
            )
            candidate_scores.append(breakdown)
            if best_breakdown is None or breakdown.total_score > best_breakdown.total_score:
                best_action = candidate
                best_delta_g = delta_g
                best_breakdown = breakdown

        return DecisionResult(
            action=best_action,
            proposed_delta_g=best_delta_g.copy(),
            chosen_score=best_breakdown,
            candidate_scores=tuple(candidate_scores),
        )
