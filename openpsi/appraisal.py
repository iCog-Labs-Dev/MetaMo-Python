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
        trans_scale = np.exp(g_trans - 0.5)
        ind_scale = np.exp(g_ind - 0.5)
        benign_novelty = stimulus.novelty * (1.0 - stimulus.risk)
        demanding_context = (stimulus.effort + stimulus.risk) / 2.0
        
        # 2. Calculate appraisal-driven updates per modulator.
        # Novel but low-risk inputs should feel energizing and attractive;
        # risky or demanding inputs should raise caution and processing depth.
        delta_valence = (
            0.75 * stimulus.conduciveness
            + 0.25 * benign_novelty
            - 0.55 * stimulus.risk
            - 0.15 * stimulus.effort
        )
        delta_arousal = (
            stimulus.novelty * (1.0 + 0.5 * arousal_feedback)
            + 0.15 * stimulus.risk
            - 0.35 * stimulus.effort
        )
        delta_approach = (
            0.65 * benign_novelty
            + 0.35 * stimulus.conduciveness
            - 0.75 * stimulus.risk
        )
        delta_resolution = (
            0.55 * stimulus.conduciveness
            + 0.35 * stimulus.effort
            + 0.20 * stimulus.risk
        )
        delta_threshold = (
            0.70 * stimulus.risk
            + 0.25 * demanding_context
            - 0.15 * stimulus.conduciveness
        )
        delta_securing = (
            0.80 * stimulus.risk
            + 0.20 * stimulus.effort
            - 0.30 * benign_novelty
            - 0.10 * stimulus.conduciveness
        )
        
        # 3. Apply MAGUS Overgoal Scaling
        # g_over^Trans scales up arousal and approach to encourage adaptive risk-taking,
        # while g_over^Ind scales up threshold and securing to preserve safety.
        M_prime[M_AROUSAL] += delta_arousal * trans_scale
        M_prime[M_APPROACH] += delta_approach * trans_scale
        M_prime[M_THRESHOLD] += delta_threshold * ind_scale
        M_prime[M_SECURING] += delta_securing * ind_scale
        M_prime[M_VALENCE] += delta_valence
        M_prime[M_RESOLUTION] += delta_resolution

        # 4. Bound the modulators to ensure they stay within [0, 1] mathematically.
        M_prime = 1.0 / (1.0 + np.exp(-4.0 * (M_prime - 0.5)))
        
        # Return a new state object with the unchanged G and the newly updated M'
        return MotivationalState(G=state.G.copy(), M=M_prime)