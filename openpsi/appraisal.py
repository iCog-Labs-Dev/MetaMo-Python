import numpy as np

# Assuming these are available in your python path
from core.state import MotivationalState, Stimulus
from core.config import (
    G_IND, 
    G_TRANS,
    M_VALENCE, 
    M_AROUSAL, 
    M_APPROACH, 
    M_RESOLUTION, 
    M_THRESHOLD, 
    M_SECURING
)
from category.functors import AppraisalComonad

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

class OpenPsiAppraisal(AppraisalComonad):
    """
    Implements the OpenPsi appraisal layer as the Comonad (\Psi)[cite: 49].
    Updates the six affective modulators based on external/internal stimuli[cite: 51].
    """

    def extract(self, state: MotivationalState) -> MotivationalState:
        """
        The comonadic counit. Extracts the current state.
        """
        return state

    def appraise(self, state: MotivationalState, stimulus: Stimulus) -> MotivationalState:
        """
        Applies \Psi((G, M), s) = (G, M')[cite: 55].
        Updates the modulators M based on stimulus novelty, conduciveness, and risk,
        while applying MAGUS overgoal scaling[cite: 116, 117].
        """
        # Create a copy to maintain functional purity
        M_prime = state.M.copy()
        
        # 1. Extract current MAGUS overgoals for scaling
        g_ind = state.G[G_IND]      # Individuation
        g_trans = state.G[G_TRANS]  # Transcendence
        arousal_feedback = sigmoid((state.M[M_AROUSAL] - 0.5) * 5.0)
        
        # 2. Calculate base modulator updates from the stimulus
        # OpenPsi novelty raises arousal and approach[cite: 188].
        delta_arousal = stimulus.novelty * (1.0 + 0.5 * arousal_feedback)
        delta_approach = stimulus.novelty
        
        # Goal conduciveness raises valence and resolution[cite: 188].
        delta_valence = stimulus.conduciveness
        delta_resolution = stimulus.conduciveness
        risk_spike = stimulus.risk * np.exp(g_ind * 2.0)
        
        # 3. Apply MAGUS Overgoal Scaling
        # g_over^Trans scales up arousal and approach to encourage adaptive risk-taking.
        M_prime[M_AROUSAL] += delta_arousal * np.exp(g_trans - 0.5)
        M_prime[M_APPROACH] += delta_approach * np.exp(g_trans - 0.5)
    
        M_prime[M_THRESHOLD] += risk_spike
        M_prime[M_SECURING] += risk_spike
    
        M_prime[M_VALENCE] += delta_valence
        M_prime[M_RESOLUTION] += delta_resolution
    
        M_prime[M_AROUSAL] -= stimulus.effort * 0.5
        
        
        # 5. Bound the modulators to ensure they stay within [0, 1] mathematically
        M_prime = 1.0 / (1.0 + np.exp(-4.0 * (M_prime - 0.5)))
        
        # Return a new state object with the unchanged G and the newly updated M'
        return MotivationalState(G=state.G.copy(), M=M_prime)