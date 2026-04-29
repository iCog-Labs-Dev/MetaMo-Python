import numpy as np
from core.state import MotivationalState
from core.config import (
    G_IND,
    G_TRANS,
    ALPHA_0,
    BETA_0
)

def calculate_blend_factor(state: MotivationalState) -> float:
    """
    Calculates the dynamic blend factor (\alpha) based on current overgoals.
    Formula: \alpha = \alpha_0(1 - g_over^{Ind}) + \beta_0 * g_over^{Trans}
    """
    g_ind = state.G[G_IND]
    g_trans = state.G[G_TRANS]
    
    # Individuation reduces alpha (slowing change), Transcendence increases it (speeding growth).
    alpha = ALPHA_0 * (1.0 - g_ind) + BETA_0 * g_trans
    
    # Ensure alpha remains strictly bounded between 0 and 1.
    return float(np.clip(alpha, 0.0, 1.0))

def blend_states(current_state: MotivationalState, target_state: MotivationalState, lipschitz_constant: float = 1.0, max_allowed_drift: float = 0.1,
    min_alpha_scale: float = 0.125,) -> MotivationalState:
    """
    Smoothly interpolates between the current state (x_t) and the proposed target state (x^*).
    Formula: x_{t+1} = (1 - \alpha)x_t + \alpha * x^*
    The step size is reduced automatically if the proposed blend violates the self-model drift bound.
    """
    base_alpha = calculate_blend_factor(current_state)
    alpha = base_alpha
    min_alpha = base_alpha * min_alpha_scale

    while True:
        next_state = MotivationalState(
            G=((1.0 - alpha) * current_state.G) + (alpha * target_state.G),
            M=((1.0 - alpha) * current_state.M) + (alpha * target_state.M),
        )
        if check_self_model_drift(
            current_state,
            next_state,
            lipschitz_constant=lipschitz_constant,
            max_allowed_drift=max_allowed_drift,
        ) or alpha <= min_alpha:
            return next_state
        alpha *= 0.5

def check_self_model_drift(
    current_state: MotivationalState, 
    next_state: MotivationalState, 
    lipschitz_constant: float = 1.0, 
    max_allowed_drift: float = 0.1
) -> bool:
    """
    Validates that the change in state does not shatter the agent's internal self-model.
    Based on the assumption: d_M(H(x), H(y)) <= L_H * d_X(x, y).
    """
    # Calculate the distance moved in the state space
    distance_moved = current_state.distance_to(next_state)
    
    # Approximate the drift in the self-model
    approximated_drift = lipschitz_constant * distance_moved
    
    return approximated_drift <= max_allowed_drift