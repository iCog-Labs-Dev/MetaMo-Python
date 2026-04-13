import numpy as np
from typing import List, Tuple

# Assuming these are available in your python path
from core.state import MotivationalState, Action
from core.config import (
    G_CURIO,
    G_ETHIC,
    G_HELP,
    G_IND,
    G_NOVEL,
    G_SELF,
    G_SOC, 
    G_TRANS,
    LAMBDA_IND,
    M_APPROACH,
    M_RESOLUTION,
    NUM_GOALS
)
from category.functors import DecisionMonad

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

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
        general_drive = (state.M[M_APPROACH] + state.M[M_RESOLUTION]) / 2.0
        # Determine general "drive" from the modulators to scale primary goal pursuit.
        # In a fully fleshed-out system, each primary goal would map to a specific modulator[cite: 168].
        # Here we use an average of approach and resolution as a generic drive multiplier.
        general_drive = (state.M[M_APPROACH] + state.M[M_RESOLUTION]) / 2.0
        for candidate in candidates:
            base_score = 0.0
            conflict_penalty = 0.0
            
            # 1. Hierarchical Gating: Overgoals act as activation thresholds
            # If an action promotes curiosity, but Transcendence is dead (near 0), 
            # the action score is exponentially crushed.
            trans_gate = sigmoid((g_trans - 0.3) * 10) # Drops sharply to 0 if g_trans < 0.3
            ind_gate = sigmoid((g_ind - 0.3) * 10)
            
            for i in range(2, NUM_GOALS):
                # Calculate alignment
                alignment = state.G[i] * general_drive * candidate.goal_correlations[i]
                
                # Apply gating based on goal type (simplified: assuming odds are exploratory, evens are cautious)
                if i in [G_CURIO, G_NOVEL, G_SELF]:
                    alignment *= trans_gate
                elif i in [G_HELP, G_ETHIC, G_SOC]:
                    alignment *= ind_gate
                    
                base_score += alignment
                
            # 2. Non-Linear Conflict Penalty (Dot Product Check)
            # If an action correlates positively with Curiosity (+0.8) but negatively with Ethics (-0.9),
            # the conflict magnitude spikes non-linearly.
            curio_ethic_conflict = candidate.goal_correlations[G_CURIO] * candidate.goal_correlations[G_ETHIC]
            if curio_ethic_conflict < -0.2:
                # Exponentially punish actions that pit core goals against each other
                conflict_penalty = np.exp(abs(curio_ethic_conflict) * 3.0)
            
            # 3. Dynamic Overgoal Risk Scaling
            # Risk penalty compounds if Individuation is very high
            risk_penalty = LAMBDA_IND * (candidate.risk_estimate ** (2.0 - g_ind)) 
            
            total_score = base_score - risk_penalty - conflict_penalty
            
            if total_score > best_score:
                best_score = total_score
                best_action = candidate
                
        # Update G normally, or apply similar non-linear blending
        next_G = np.clip(state.G + best_action.delta_g, 0.0, 1.0)
        return best_action, MotivationalState(G=next_G, M=state.M.copy())