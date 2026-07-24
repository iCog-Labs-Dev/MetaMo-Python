from dataclasses import dataclass

import numpy as np
from core.state import MotivationalState
from core.config import (
    ALPHA_0,
    BETA_0,
)

MIN_BLEND_ALPHA = 1e-6


def _modulator_value(state: MotivationalState, name: str, default: float = 0.0) -> float:
    try:
        return state.modulator(name)
    except KeyError:
        return default


def _summary(values: np.ndarray) -> np.ndarray:
    """
    Return a fixed-size summary for a possibly empty coordinate block.
    """
    if values.size == 0:
        return np.zeros(3)
    return np.array([
        float(np.mean(values)),
        float(np.max(values)),
        float(np.linalg.norm(values)),
    ])


@dataclass(frozen=True)
class SelfModel:
    """
    Compact explicit self-model H(x) derived from motivational state.
    """

    vector: np.ndarray

    def distance_to(self, other: "SelfModel") -> float:
        return float(np.linalg.norm(self.vector - other.vector))


@dataclass(frozen=True)
class SelfModelDriftResult:
    """
    Measures continuity of self-model between two motivational states.
    """

    current_model: SelfModel
    next_model: SelfModel
    state_distance: float
    self_model_distance: float
    lipschitz_bound: float
    combined_drift: float
    max_allowed_drift: float
    holds: bool


@dataclass(frozen=True)
class BlendResult:
    """
    Result of incremental objective embodiment.
    """

    state: MotivationalState
    alpha: float
    base_alpha: float
    drift: SelfModelDriftResult


def estimate_self_model(state: MotivationalState) -> SelfModel:
    """
    Builds an explicit lightweight self-model H(x) from schema structure.
    """
    g_ind = state.goal(state.schema.goals.individuation_name)
    g_trans = state.goal(state.schema.goals.transcendence_name)

    primary_values = state.G[
        state.schema.goals.primary_start:state.schema.goals.anti_goal_start
    ]
    anti_goal_values = state.G[state.schema.goals.anti_goal_start:]
    core_modulator_values = state.M[:state.schema.modulators.core_count]
    app_modulator_values = state.M[state.schema.modulators.core_count:]

    caution_posture = (
        _modulator_value(state, "threshold") + _modulator_value(state, "securing")
    ) / 2.0
    exploration_posture = (
        _modulator_value(state, "arousal") + _modulator_value(state, "approach")
    ) / 2.0
    clarity_posture = _modulator_value(state, "resolution")
    affect_posture = _modulator_value(state, "valence")

    return SelfModel(
        vector=np.concatenate([
            state.G,
            state.M,
            np.array([g_ind, g_trans]),
            _summary(primary_values),
            _summary(anti_goal_values),
            _summary(core_modulator_values),
            _summary(app_modulator_values),
            np.array([
                caution_posture,
                exploration_posture,
                clarity_posture,
                affect_posture,
            ]),
        ])
    )

def calculate_blend_factor(state: MotivationalState) -> float:
    """
    Calculates the dynamic blend factor (α) based on current overgoals.
    Formula: α = α_0(1 - g_over^{Ind}) + β_0 * g_over^{Trans}
    """
    g_ind = state.goal(state.schema.goals.individuation_name)
    g_trans = state.goal(state.schema.goals.transcendence_name)
    
    # Individuation reduces alpha (slowing change), Transcendence increases it (speeding growth).
    alpha = ALPHA_0 * (1.0 - g_ind) + BETA_0 * g_trans
    
    # Ensure alpha remains strictly bounded between 0 and 1.
    return float(np.clip(alpha, MIN_BLEND_ALPHA, 1.0 - MIN_BLEND_ALPHA))

def measure_blend(
    current_state: MotivationalState,
    target_state: MotivationalState,
    lipschitz_constant: float = 1.0,
    max_allowed_drift: float = 0.1,
    min_alpha_scale: float = 0.125,
    state_drift_weight: float = 0.5,
    self_model_drift_weight: float = 0.5,
) -> BlendResult:
    """
    Measures and applies the incremental embodiment update.
    """
    base_alpha = calculate_blend_factor(current_state)
    alpha = base_alpha
    min_alpha = base_alpha * min_alpha_scale

    while True:
        next_state = MotivationalState(
            G=((1.0 - alpha) * current_state.G) + (alpha * target_state.G),
            M=((1.0 - alpha) * current_state.M) + (alpha * target_state.M),
            schema=current_state.schema,
        )
        drift = measure_self_model_drift(
            current_state,
            next_state,
            lipschitz_constant=lipschitz_constant,
            max_allowed_drift=max_allowed_drift,
            state_drift_weight=state_drift_weight,
            self_model_drift_weight=self_model_drift_weight,
        )
        if drift.holds or alpha <= min_alpha:
            return BlendResult(
                state=next_state,
                alpha=alpha,
                base_alpha=base_alpha,
                drift=drift,
            )
        alpha *= 0.5


def blend_states(
    current_state: MotivationalState,
    target_state: MotivationalState,
    lipschitz_constant: float = 1.0,
    max_allowed_drift: float = 0.1,
    min_alpha_scale: float = 0.125,
) -> MotivationalState:
    """
    Smoothly interpolates between the current state (x_t) and the proposed target state (x^*).
    Formula: x_{t+1} = (1 - α)x_t + α * x^*
    The step size is reduced automatically if the proposed blend violates the self-model drift bound.
    """
    return measure_blend(
        current_state,
        target_state,
        lipschitz_constant=lipschitz_constant,
        max_allowed_drift=max_allowed_drift,
        min_alpha_scale=min_alpha_scale,
    ).state

def measure_self_model_drift(
    current_state: MotivationalState, 
    next_state: MotivationalState, 
    lipschitz_constant: float = 1.0, 
    max_allowed_drift: float = 0.1,
    state_drift_weight: float = 0.5,
    self_model_drift_weight: float = 0.5,
) -> SelfModelDriftResult:
    """
    Measures explicit self-model drift alongside state-space drift.
    """
    distance_moved = current_state.distance_to(next_state)
    current_model = estimate_self_model(current_state)
    next_model = estimate_self_model(next_state)
    self_model_distance = current_model.distance_to(next_model)
    lipschitz_bound = lipschitz_constant * distance_moved
    combined_drift = (
        state_drift_weight * lipschitz_bound
        + self_model_drift_weight * self_model_distance
    )

    return SelfModelDriftResult(
        current_model=current_model,
        next_model=next_model,
        state_distance=distance_moved,
        self_model_distance=self_model_distance,
        lipschitz_bound=lipschitz_bound,
        combined_drift=combined_drift,
        max_allowed_drift=max_allowed_drift,
        holds=combined_drift <= max_allowed_drift,
    )


def check_self_model_drift(
    current_state: MotivationalState,
    next_state: MotivationalState,
    lipschitz_constant: float = 1.0,
    max_allowed_drift: float = 0.1,
) -> bool:
    """
    Validates that the change in state does not shatter the agent's internal self-model.
    """
    return measure_self_model_drift(
        current_state,
        next_state,
        lipschitz_constant=lipschitz_constant,
        max_allowed_drift=max_allowed_drift,
    ).holds


class DefaultCoherencePolicy:
    """
    Default self-model and incremental embodiment policy.
    """

    def estimate_self_model(self, state: MotivationalState) -> SelfModel:
        return estimate_self_model(state)

    def calculate_blend_factor(self, state: MotivationalState) -> float:
        return calculate_blend_factor(state)

    def measure_self_model_drift(
        self,
        current_state: MotivationalState,
        next_state: MotivationalState,
        lipschitz_constant: float = 1.0,
        max_allowed_drift: float = 0.1,
        state_drift_weight: float = 0.5,
        self_model_drift_weight: float = 0.5,
    ) -> SelfModelDriftResult:
        distance_moved = current_state.distance_to(next_state)
        current_model = self.estimate_self_model(current_state)
        next_model = self.estimate_self_model(next_state)
        self_model_distance = current_model.distance_to(next_model)
        lipschitz_bound = lipschitz_constant * distance_moved
        combined_drift = (
            state_drift_weight * lipschitz_bound
            + self_model_drift_weight * self_model_distance
        )

        return SelfModelDriftResult(
            current_model=current_model,
            next_model=next_model,
            state_distance=distance_moved,
            self_model_distance=self_model_distance,
            lipschitz_bound=lipschitz_bound,
            combined_drift=combined_drift,
            max_allowed_drift=max_allowed_drift,
            holds=combined_drift <= max_allowed_drift,
        )

    def measure_blend(
        self,
        current_state: MotivationalState,
        target_state: MotivationalState,
        lipschitz_constant: float = 1.0,
        max_allowed_drift: float = 0.1,
        min_alpha_scale: float = 0.125,
        state_drift_weight: float = 0.5,
        self_model_drift_weight: float = 0.5,
    ) -> BlendResult:
        base_alpha = self.calculate_blend_factor(current_state)
        alpha = base_alpha
        min_alpha = base_alpha * min_alpha_scale

        while True:
            next_state = MotivationalState(
                G=((1.0 - alpha) * current_state.G) + (alpha * target_state.G),
                M=((1.0 - alpha) * current_state.M) + (alpha * target_state.M),
                schema=current_state.schema,
            )
            drift = self.measure_self_model_drift(
                current_state,
                next_state,
                lipschitz_constant=lipschitz_constant,
                max_allowed_drift=max_allowed_drift,
                state_drift_weight=state_drift_weight,
                self_model_drift_weight=self_model_drift_weight,
            )
            if drift.holds or alpha <= min_alpha:
                return BlendResult(
                    state=next_state,
                    alpha=alpha,
                    base_alpha=base_alpha,
                    drift=drift,
                )
            alpha *= 0.5

    def blend_states(
        self,
        current_state: MotivationalState,
        target_state: MotivationalState,
        lipschitz_constant: float = 1.0,
        max_allowed_drift: float = 0.1,
        min_alpha_scale: float = 0.125,
    ) -> MotivationalState:
        return self.measure_blend(
            current_state,
            target_state,
            lipschitz_constant=lipschitz_constant,
            max_allowed_drift=max_allowed_drift,
            min_alpha_scale=min_alpha_scale,
        ).state

    def check_self_model_drift(
        self,
        current_state: MotivationalState,
        next_state: MotivationalState,
        lipschitz_constant: float = 1.0,
        max_allowed_drift: float = 0.1,
    ) -> bool:
        return self.measure_self_model_drift(
            current_state,
            next_state,
            lipschitz_constant=lipschitz_constant,
            max_allowed_drift=max_allowed_drift,
        ).holds
