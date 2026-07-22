from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from core.schema import MotivationSchema
from core.state import Action, MotivationalState


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def positive_part(value: float) -> float:
    return max(0.0, value)


def same_coordinate_layout(left: MotivationSchema, right: MotivationSchema) -> bool:
    return (
        left.goal_names == right.goal_names
        and left.modulator_names == right.modulator_names
    )


@dataclass(frozen=True)
class DecisionProfile:
    """
    Application-specific MAGUS decision semantics.

    The generic MAGUS decision monad owns the default weighted-additive DS
    formula. A profile supplies coordinate meanings, modulator couplings,
    compatibility factors, and anti-goal penalty weights.
    """

    name: str
    schema: MotivationSchema
    goal_modulators: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    individuation_goal_names: tuple[str, ...] = ()
    transcendence_goal_names: tuple[str, ...] = ()
    balanced_goal_names: tuple[str, ...] = ()
    anti_goal_penalty_weights: Mapping[str, float] = field(default_factory=dict)

    def validate(self, state: MotivationalState, candidate: Action) -> None:
        if not same_coordinate_layout(state.schema, self.schema):
            raise ValueError(f"state schema does not match decision profile {self.name}")
        if not same_coordinate_layout(candidate.schema, state.schema):
            raise ValueError("candidate schema must match state schema")

    def primary_goal_indices(self) -> range:
        return range(self.schema.goals.primary_start, self.schema.goals.anti_goal_start)

    def anti_goal_indices(self) -> range:
        return range(self.schema.goals.anti_goal_start, self.schema.num_goals)

    def scored_goal_indices(self) -> range:
        """Compatibility alias for the primary-goal part of the DS formula."""
        return self.primary_goal_indices()

    def relevant_modulator(self, state: MotivationalState, goal_idx: int) -> float:
        goal_name = self.schema.goal_names[goal_idx]
        modulator_names = self.goal_modulators.get(goal_name, ())
        if not modulator_names:
            return 1.0
        return float(np.mean([state.modulator(name) for name in modulator_names]))

    def overgoal_support(self, goal_idx: int, g_ind: float, g_trans: float) -> float:
        """
        Default profile-level compatibility factor driven by the two overgoals.
        """
        goal_name = self.schema.goal_names[goal_idx]
        ind_support = sigmoid((g_ind - 0.5) * 6.0)
        trans_support = sigmoid((g_trans - 0.5) * 6.0)

        if goal_name in self.individuation_goal_names:
            return 0.5 + 0.5 * ind_support
        if goal_name in self.transcendence_goal_names:
            return 0.5 + 0.5 * trans_support
        if goal_name in self.balanced_goal_names:
            return 0.5 + 0.25 * (ind_support + trans_support)
        return 1.0

    def compatibility_factor(
        self,
        goal_idx: int,
        state: MotivationalState,
        candidate: Action,
    ) -> float:
        """
        Profile-defined kappa_i(x) / MIC factor used inside f.
        """
        g_ind = state.goal(self.schema.goals.individuation_name)
        g_trans = state.goal(self.schema.goals.transcendence_name)
        return self.overgoal_support(goal_idx, g_ind, g_trans)

    def f(self, goal_idx: int, state: MotivationalState, candidate: Action) -> float:
        """
        Expanded f(g_i, M_k, MIC) term:
        g_i * m_{rho(i)} * kappa_i(x) * corr_i(a).
        """
        return float(
            state.G[goal_idx]
            * self.relevant_modulator(state, goal_idx)
            * self.compatibility_factor(goal_idx, state, candidate)
            * candidate.goal_correlations[goal_idx]
        )

    def anti_goal_penalty(self, state: MotivationalState, candidate: Action) -> float:
        """
        Sum lambda_{a_j} * a_j * corr_{a_j}(a) over anti-goal coordinates.

        Anti-goal correlations are interpreted as activation pressure, so the
        default clips negative values to zero instead of treating them as reward.
        Applications can override this method if they need signed anti-goal
        correlations.
        """
        penalty = 0.0
        for goal_idx in self.anti_goal_indices():
            anti_goal_name = self.schema.goal_names[goal_idx]
            weight = self.anti_goal_penalty_weights.get(anti_goal_name, 1.0)
            activation = positive_part(candidate.goal_correlations[goal_idx])
            penalty += weight * state.G[goal_idx] * activation
        return float(penalty)

    def decision_score(
        self,
        state: MotivationalState,
        candidate: Action,
        lambda_ind: float,
        lambda_trans: float,
    ) -> float:
        """
        Default MAGUS decision score:

        DS(a) =
            f(g_i, M_k, MIC)
            - lambda_Ind g_over^Ind
            + lambda_Trans g_over^Trans
            - anti-goal penalty
        """
        g_ind = state.goal(self.schema.goals.individuation_name)
        g_trans = state.goal(self.schema.goals.transcendence_name)
        primary_score = sum(
            self.f(goal_idx, state, candidate)
            for goal_idx in self.primary_goal_indices()
        )
        return float(
            primary_score
            - (lambda_ind * g_ind)
            + (lambda_trans * g_trans)
            - self.anti_goal_penalty(state, candidate)
        )

    def score_candidate(
        self,
        state: MotivationalState,
        candidate: Action,
        lambda_ind: float,
        lambda_trans: float,
    ) -> float:
        self.validate(state, candidate)
        return self.decision_score(state, candidate, lambda_ind, lambda_trans)
