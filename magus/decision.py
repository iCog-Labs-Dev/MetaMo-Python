import numpy as np
from typing import List, Tuple

# Assuming these are available in your python path
from core.state import MotivationalState, Action
from core.config import (
    G_IND, 
    G_TRANS,
    LAMBDA_IND,
    LAMBDA_TRANS,
    M_APPROACH,
    M_RESOLUTION,
    NUM_GOALS
)
from category.functors import DecisionMonad

class MagusDecision(DecisionMonad):
    """
    Implements the MAGUS hierarchical decision layer as the Monad (\mathbb{D}).
    Scores candidate actions using the updated emotional modulators and dual overgoals,
    then updates the goal vector G.
    """

    def unit(self, state: MotivationalState) -> MotivationalState:
        """
        The monadic unit (\eta). 
        Injects the state into the monadic context without altering it.
        """
        return state

    def decide(self, state: MotivationalState, candidates: List[Action]) -> Tuple[Action, MotivationalState]:
        """
        Applies \mathbb{D}((G, M)) = (G + \Delta G, M).
        Scores each candidate action, selects the highest-scoring one, 
        and updates the goal vector.
        """
        if not candidates:
            raise ValueError("Must provide at least one candidate action to the decision monad.")

        best_action = None
        best_score = -float('inf')
        
        # Extract the dual overgoals that govern decision constraints
        g_ind = state.G[G_IND]      # Individuation: enforces safety and caution
        g_trans = state.G[G_TRANS]  # Transcendence: encourages growth and exploration
        
        # Determine general "drive" from the modulators to scale primary goal pursuit.
        # In a fully fleshed-out system, each primary goal would map to a specific modulator[cite: 168].
        # Here we use an average of approach and resolution as a generic drive multiplier.
        general_drive = (state.M[M_APPROACH] + state.M[M_RESOLUTION]) / 2.0

        for candidate in candidates:
            # 1. Base Score: Sum of (goal_intensity * modulator * action_correlation)
            # We skip the first two indices (0 and 1) as they are the overgoals.
            base_score = 0.0
            for i in range(2, NUM_GOALS):
                base_score += state.G[i] * general_drive * candidate.goal_correlations[i]
            
            # 2. Individuation Penalty: Suppress actions with high risk when self-preservation is critical[cite: 189].
            # Formula: \lambda_{Ind} * g_{over}^{Ind} * risk(a)
            risk_penalty = LAMBDA_IND * g_ind * candidate.risk_estimate
            
            # 3. Transcendence Reward: Boost actions to favor exploratory/growth subgoals[cite: 210].
            # Formula: \lambda_{Trans} * g_{over}^{Trans}
            growth_reward = LAMBDA_TRANS * g_trans
            
            # 4. Total Score Calculation
            total_score = base_score - risk_penalty + growth_reward
            
            if total_score > best_score:
                best_score = total_score
                best_action = candidate
                
        # Update the goal vector using the selected action's \Delta G 
        # This boosts those goals most in need while respecting the constraints[cite: 170].
        next_G = state.G + best_action.delta_g
        
        # Ensure goal intensities remain mathematically bounded between 0 and 1
        next_G = np.clip(next_G, 0.0, 1.0)
        
        # Return the chosen action and the new state (modulators M remain unchanged by \mathbb{D})
        return best_action, MotivationalState(G=next_G, M=state.M.copy())