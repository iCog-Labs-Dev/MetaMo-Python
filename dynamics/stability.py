import numpy as np
from typing import List
from core.state import MotivationalState
from core.config import (
    G_IND, 
    THETA_SAFE, 
    G_MAX, 
    ETA_BOUNDARY, 
    C_CONTRACT, 
    EPSILON,
    M_SECURING,
    M_THRESHOLD
)
from core.state import Stimulus, Action

def is_in_safe_region(state: MotivationalState) -> bool:
    """
    Checks if the state is within the designated safe region R.
    R = {(G, M) | g_over^Ind >= \theta_{safe} \wedge ||G|| <= G_{max}}[cite: 131, 174].
    """
    g_ind = state.G[G_IND]
    g_norm = np.linalg.norm(state.G)
    
    return (g_ind >= THETA_SAFE) and (g_norm <= G_MAX)

def boundary_pressure(state: MotivationalState) -> float:
    """
    Returns how strongly the safety boundary should constrain updates.
    0.0 means comfortably inside the safe region; 1.0 means outside or on the edge.
    """
    if not is_in_safe_region(state):
        return 1.0

    dist_to_boundary = distance_to_unsafe_boundary(state)
    if dist_to_boundary >= ETA_BOUNDARY:
        return 0.0

    return float(np.clip(1.0 - (dist_to_boundary / ETA_BOUNDARY), 0.0, 1.0))

def distance_to_unsafe_boundary(state: MotivationalState) -> float:
    """
    Approximates the distance from the current state to the edge of the safe region (\partial R).
    Calculates how close the agent is to violating THETA_SAFE or G_MAX.
    """
    # Distance to the individuation safety floor
    dist_to_theta = max(0.0, state.G[G_IND] - THETA_SAFE)
    
    # Distance to the maximum goal norm ceiling
    g_norm = np.linalg.norm(state.G)
    dist_to_g_max = max(0.0, G_MAX - g_norm)
    
    # The actual distance to the boundary is determined by whichever constraint is closer
    return min(dist_to_theta, dist_to_g_max)

def is_in_boundary_band(state: MotivationalState) -> bool:
    """
    Checks if the state is in the boundary band B_\eta.
    B_\eta = {x \in R | dist(x, X \setminus R) <= \eta}[cite: 383].
    """
    if not is_in_safe_region(state):
        return False # It is already outside the safe region entirely
        
    dist_to_boundary = distance_to_unsafe_boundary(state)
    return dist_to_boundary <= ETA_BOUNDARY

def raise_boundary_caution(state: MotivationalState) -> MotivationalState:
    """
    Raises caution-related modulators as the state approaches or crosses the safety boundary.
    This mirrors the paper's boundary-sensitive appraisal before decision.
    """
    pressure = boundary_pressure(state)
    if pressure == 0.0:
        return state

    next_state = state.copy()
    caution_boost = 0.25 * pressure
    next_state.M[M_SECURING] = min(1.0, next_state.M[M_SECURING] + caution_boost)
    next_state.M[M_THRESHOLD] = min(1.0, next_state.M[M_THRESHOLD] + caution_boost)
    return next_state

def check_contractive_update_law(
    bimonad,
    x: MotivationalState, 
    y: MotivationalState, 
    stimulus: Stimulus,
    candidates: List[Action]
) -> bool:
    """
    Validates that the pseudo-bimonad update F = D \circ \Psi is contractive near the boundary.
    Requirement: d(F(x), F(y)) <= c * d(x,y) + \epsilon where c < 1[cite: 132, 176, 384].
    This ensures that high individuation near the boundary induces contraction toward safety[cite: 133].
    """
    # If neither state is in the boundary band, the contractivity constraint relaxes[cite: 134, 385].
    if not (is_in_boundary_band(x) or is_in_boundary_band(y)):
        return True # Dynamics are allowed to be flexible deep inside R[cite: 385, 403].

    # Calculate initial distance d(x, y)
    dist_initial = x.distance_to(y)
    
    # Apply the F operator to both states
    _, F_x = bimonad._compute_transition(x, stimulus, candidates)
    _, F_y = bimonad._compute_transition(y, stimulus, candidates)
    
    # Calculate final distance d(F(x), F(y))
    dist_final = F_x.distance_to(F_y)
    
    # Verify the contractive bound
    return dist_final <= (C_CONTRACT * dist_initial) + EPSILON


def apply_homeostatic_damping(state: MotivationalState, delta_g: np.ndarray) -> np.ndarray:
    """
    Actively enforces Principle 4 by damping goal updates near the boundary.
    """
    pressure = boundary_pressure(state)
    if pressure == 0.0:
        return delta_g

    # Stronger boundary pressure and higher individuation induce more contraction.
    damping_factor = max(0.0, 1.0 - (pressure * state.G[G_IND]))
    return delta_g * damping_factor


def project_to_safe_region(state: MotivationalState) -> MotivationalState:
    """
    Projects a state back into the designated safe region by restoring the individuation floor
    and shrinking the goal vector if it exceeds the allowed norm.
    """
    next_state = state.copy()
    next_state.G[G_IND] = max(next_state.G[G_IND], THETA_SAFE)

    other_idx = [idx for idx in range(next_state.G.shape[0]) if idx != G_IND]
    other_goals = next_state.G[other_idx]
    other_norm = np.linalg.norm(other_goals)
    max_other_norm = np.sqrt(max(0.0, G_MAX**2 - next_state.G[G_IND] ** 2))
    if other_norm > max_other_norm and other_norm > 0.0:
        next_state.G[other_idx] = other_goals * (max_other_norm / other_norm)

    next_state.M[M_SECURING] = min(1.0, next_state.M[M_SECURING] + 0.1)
    next_state.M[M_THRESHOLD] = min(1.0, next_state.M[M_THRESHOLD] + 0.1)
    return next_state
