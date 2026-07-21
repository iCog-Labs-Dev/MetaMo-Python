from dataclasses import dataclass

import numpy as np
from core.state import MotivationalState
from core.config import (
    G_CURIO,
    G_ETHIC,
    G_HELP,
    G_IND,
    G_NOVEL,
    G_SELF,
    G_SOC,
    G_TRANS,
    ALPHA_0,
    BETA_0,
    M_APPROACH,
    M_AROUSAL,
    M_RESOLUTION,
    M_SECURING,
    M_THRESHOLD,
    M_VALENCE,
)

MIN_BLEND_ALPHA = 1e-6


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
    Builds an explicit lightweight self-model H(x).

    The vector keeps the full motivational state plus summary commitments:
    safety, growth, service, ethics, sociality, caution, exploration, and affect.
    """
    safety_commitment = (state.G[G_IND] + state.G[G_ETHIC]) / 2.0
    growth_commitment = (state.G[G_TRANS] + state.G[G_CURIO] + state.G[G_NOVEL]) / 3.0
    service_commitment = state.G[G_HELP]
    self_commitment = state.G[G_SELF]
    social_commitment = state.G[G_SOC]
    caution_posture = (state.M[M_THRESHOLD] + state.M[M_SECURING]) / 2.0
    exploration_posture = (state.M[M_AROUSAL] + state.M[M_APPROACH]) / 2.0
    clarity_posture = state.M[M_RESOLUTION]
    affect_posture = state.M[M_VALENCE]

    return SelfModel(
        vector=np.concatenate([
            state.G,
            state.M,
            np.array([
                safety_commitment,
                growth_commitment,
                service_commitment,
                self_commitment,
                social_commitment,
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
    g_ind = state.G[G_IND]
    g_trans = state.G[G_TRANS]
    
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
