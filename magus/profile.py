from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from core.schema import MotivationSchema
from core.state import Action, MotivationalState
from magus.correlation import GoalCompatibilityMatrix
from magus.goal_change import DefaultGoalChangeCalculator
from core.decision import CandidateScoreBreakdown


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
    """

    name: str
    schema: MotivationSchema
    goal_modulators: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    individuation_goal_names: tuple[str, ...] = ()
    transcendence_goal_names: tuple[str, ...] = ()
    balanced_goal_names: tuple[str, ...] = ()
    anti_goal_penalty_weights: Mapping[str, float] = field(default_factory=dict)
    overgoal_target_features: Mapping[str, str] = field(default_factory=dict)
    anti_goal_target_features: Mapping[str, str] = field(default_factory=dict)
    compatibility_matrix: GoalCompatibilityMatrix | None = None
    overgoal_delta_scale: float = 0.02
    anti_goal_delta_scale: float = 0.03
    delta_scale: float = 0.05
    candidate_delta_weight: float = 0.0
    expected_goal_change_weight: float = 0.1
    max_goal_delta: float = 0.1

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
        mic_factor = 1.0
        if self.compatibility_matrix is not None:
            mic_factor = self.compatibility_matrix.factor_for(goal_idx, state)
        return self.overgoal_support(goal_idx, g_ind, g_trans) * mic_factor

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
        Compute the penalty for selecting an action that activates anti-goals.
        """
        penalty = 0.0
        for goal_idx in self.anti_goal_indices():
            anti_goal_name = self.schema.goal_names[goal_idx]
            weight = self.anti_goal_penalty_weights.get(anti_goal_name, 1.0)
            activation = positive_part(candidate.goal_correlations[goal_idx])
            penalty += weight * state.G[goal_idx] * activation
        return float(penalty)

    def action_risk(self, candidate: Action) -> float:
        """Return a candidate-dependent risk estimate in [0, 1]."""
        if "risk" in candidate.metadata:
            return float(np.clip(float(candidate.metadata["risk"]), 0.0, 1.0))
        ind_idx = self.schema.goal_index(self.schema.goals.individuation_name)
        ind_alignment = float(np.clip(candidate.goal_correlations[ind_idx], -1.0, 1.0))
        return (1.0 - ind_alignment) / 2.0

    def action_opportunity(self, candidate: Action) -> float:
        """Return a candidate-dependent growth opportunity estimate in [0, 1]."""
        if "opportunity" in candidate.metadata:
            return float(np.clip(float(candidate.metadata["opportunity"]), 0.0, 1.0))

        indices = [
            self.schema.goal_index(name)
            for name in self.transcendence_goal_names
            if name in self.schema.goal_names
        ]
        if indices:
            return float(np.clip(np.mean([
                positive_part(candidate.goal_correlations[idx])
                for idx in indices
            ]), 0.0, 1.0))

        trans_idx = self.schema.goal_index(self.schema.goals.transcendence_name)
        trans_alignment = float(np.clip(candidate.goal_correlations[trans_idx], -1.0, 1.0))
        return (1.0 + trans_alignment) / 2.0

    def expected_goal_change_value(
        self,
        state: MotivationalState,
        delta_g: np.ndarray,
    ) -> float:
        """Value expected progress while treating anti-goal growth as harmful."""
        value = 0.0
        for idx in range(self.schema.goals.anti_goal_start):
            value += state.G[idx] * delta_g[idx]
        for idx in self.anti_goal_indices():
            value -= state.G[idx] * delta_g[idx]
        return float(self.expected_goal_change_weight * value)

    def score_breakdown(
        self,
        state: MotivationalState,
        candidate: Action,
        delta_g: np.ndarray,
        lambda_ind: float,
        lambda_trans: float,
    ) -> CandidateScoreBreakdown:
        """Return the complete, candidate-dependent MAGUS score."""
        self.validate(state, candidate)
        primary = {
            self.schema.goal_names[idx]: self.f(idx, state, candidate)
            for idx in self.primary_goal_indices()
        }
        anti = {}
        for idx in self.anti_goal_indices():
            name = self.schema.goal_names[idx]
            weight = self.anti_goal_penalty_weights.get(name, 1.0)
            anti[name] = float(
                weight
                * state.G[idx]
                * positive_part(candidate.goal_correlations[idx])
            )

        g_ind = state.goal(self.schema.goals.individuation_name)
        g_trans = state.goal(self.schema.goals.transcendence_name)
        individuation_adjustment = -lambda_ind * g_ind * self.action_risk(candidate)
        transcendence_adjustment = (
            lambda_trans * g_trans * self.action_opportunity(candidate)
        )
        expected_change = self.expected_goal_change_value(state, delta_g)
        total = (
            sum(primary.values())
            - sum(anti.values())
            + individuation_adjustment
            + transcendence_adjustment
            + expected_change
        )
        return CandidateScoreBreakdown(
            action_id=candidate.id,
            primary_goal_contributions=primary,
            anti_goal_penalties=anti,
            individuation_adjustment=float(individuation_adjustment),
            transcendence_adjustment=float(transcendence_adjustment),
            expected_goal_change=float(expected_change),
            total_score=float(total),
        )

    def decision_score(
        self,
        state: MotivationalState,
        candidate: Action,
        lambda_ind: float,
        lambda_trans: float,
    ) -> float:
        """Compatibility helper for scoring without external feedback."""
        delta = self.delta_g(state, candidate, lambda_ind, lambda_trans)
        return self.score_breakdown(
            state, candidate, delta, lambda_ind, lambda_trans
        ).total_score

    def goal_update_value(
        self,
        goal_idx: int,
        state: MotivationalState,
        candidate: Action,
        lambda_ind: float,
        lambda_trans: float,
    ) -> float:
        """
        Additive update for one primary goal coordinate
        """
        g_ind = state.goal(self.schema.goals.individuation_name)
        g_trans = state.goal(self.schema.goals.transcendence_name)
        return float(
            self.f(goal_idx, state, candidate)
            - (lambda_ind * g_ind)
            + (lambda_trans * g_trans)
        )

    def delta_g(
        self,
        state: MotivationalState,
        candidate: Action,
        lambda_ind: float,
        lambda_trans: float,
        feedback: Any = None,
    ) -> np.ndarray:
        """
        Compute the bounded proposed Delta G(a) for the selected candidate.
        """
        return DefaultGoalChangeCalculator(self).delta_g(
            state,
            candidate,
            feedback=feedback,
            lambda_ind=lambda_ind,
            lambda_trans=lambda_trans,
        )

    def aggregate(
        self,
        delta_g: np.ndarray,
        state: MotivationalState,
        candidate: Action,
        lambda_ind: float,
        lambda_trans: float,
    ) -> float:
        """
        Application aggregation A(Delta G(a), x).
        """
        return self.score_breakdown(
            state,
            candidate,
            delta_g,
            lambda_ind,
            lambda_trans,
        ).total_score

    def score_candidate(
        self,
        state: MotivationalState,
        candidate: Action,
        lambda_ind: float,
        lambda_trans: float,
    ) -> float:
        self.validate(state, candidate)
        delta = self.delta_g(state, candidate, lambda_ind, lambda_trans)
        return self.aggregate(delta, state, candidate, lambda_ind, lambda_trans)
