import numpy as np
from core.state import MotivationalState
from core.config import (
    G_IND, 
    THETA_SAFE, 
    G_MAX, 
    ETA_BOUNDARY, 
    C_CONTRACT, 
    EPSILON
)
# Assuming bimonad is available
from category.bimonad import MetaMoPseudoBimonad
from core.state import Stimulus, Action
from typing import List

def is_in_safe_region(state: MotivationalState) -> bool:
    """
    Checks if the state is within the designated safe region R.
    R = {(G, M) | g_over^Ind >= \theta_{safe} \wedge ||G|| <= G_{max}}[cite: 131, 174].
    """
    g_ind = state.G[G_IND]
    g_norm = np.linalg.norm(state.G)
    
    return (g_ind >= THETA_SAFE) and (g_norm <= G_MAX)

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

def check_contractive_update_law(
    bimonad: MetaMoPseudoBimonad, 
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
    _, F_x = bimonad.step(x, stimulus, candidates)
    _, F_y = bimonad.step(y, stimulus, candidates)
    
    # Calculate final distance d(F(x), F(y))
    dist_final = F_x.distance_to(F_y)
    
    # Verify the contractive bound
    return dist_final <= (C_CONTRACT * dist_initial) + EPSILON