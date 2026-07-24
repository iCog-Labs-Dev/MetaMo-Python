from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.features import GoalChangeFeedback
from core.state import Action, MotivationalState


class GoalChangeCalculator(ABC):
    """
    Computes the proposed goal-vector change after an action is selected.
    """

    @abstractmethod
    def delta_g(
        self,
        state: MotivationalState,
        candidate: Action,
        feedback: Any = None,
        *,
        lambda_ind: float,
        lambda_trans: float,
    ) -> np.ndarray:
        pass


@dataclass(frozen=True)
class DefaultGoalChangeCalculator(GoalChangeCalculator):
    """
    Default weighted-additive primary-goal update.
    """

    profile: Any

    def _apply_overgoal_targets(
        self,
        delta: np.ndarray,
        state: MotivationalState,
        feedback: GoalChangeFeedback | None,
    ) -> None:
        if not isinstance(feedback, GoalChangeFeedback):
            return

        for overgoal_name in (
            self.profile.schema.goals.individuation_name,
            self.profile.schema.goals.transcendence_name,
        ):
            feature_name = self.profile.overgoal_target_features.get(overgoal_name)
            if not feature_name or feature_name not in feedback.features:
                continue
            goal_idx = self.profile.schema.goal_index(overgoal_name)
            target = feedback.numeric(feature_name)
            delta[goal_idx] = self.profile.overgoal_delta_scale * (
                target - state.G[goal_idx]
            )

    def delta_g(
        self,
        state: MotivationalState,
        candidate: Action,
        feedback: Any = None,
        *,
        lambda_ind: float,
        lambda_trans: float,
    ) -> np.ndarray:
        self.profile.validate(state, candidate)
        delta = np.zeros(self.profile.schema.num_goals, dtype=float)

        for goal_idx in self.profile.primary_goal_indices():
            delta[goal_idx] = self.profile.delta_scale * self.profile.goal_update_value(
                goal_idx,
                state,
                candidate,
                lambda_ind,
                lambda_trans,
            )

        self._apply_overgoal_targets(delta, state, feedback)

        if self.profile.candidate_delta_weight:
            delta += self.profile.candidate_delta_weight * candidate.delta_g

        return np.clip(
            delta,
            -self.profile.max_goal_delta,
            self.profile.max_goal_delta,
        )
