from typing import List, Tuple

# Assuming these are available in your python path
from core.state import MotivationalState, Stimulus, Action
from core.config import LAX_DISTRIBUTIVE_DELTA
from category.functors import AppraisalComonad, DecisionMonad

class MetaMoPseudoBimonad:
    """
    Represents the composite appraisal-then-decision operator F = D \circ \Psi.
    This forms a pseudo-bimonad on the motivational state space X = G \times M[cite: 28, 309].
    """
    def __init__(self, appraisal: AppraisalComonad, decision: DecisionMonad):
        self.appraisal = appraisal
        self.decision = decision

    def step(self, state: MotivationalState, stimulus: Stimulus, candidates: List[Action]) -> Tuple[Action, MotivationalState]:
        """
        Executes one full cycle of F = D \circ \Psi.
        This governs the motivational coalgebra \alpha: X \to F(X)[cite: 310, 316].
        
        Args:
            state: The current motivational state x_t.
            stimulus: The perceived environment state s_t.
            candidates: Available actions for the decision monad.
            
        Returns:
            The chosen action a_t and the next state x_{t+1}.
        """
        # 1. Appraise (\Psi) - Update modulators based on stimulus[cite: 314].
        appraised_state = self.appraisal.appraise(state, stimulus)
        
        # 2. Decide (\mathbb{D}) - Score candidates and update goals[cite: 315].
        chosen_action, next_state = self.decision.decide(appraised_state, candidates)
        
        return chosen_action, next_state

    def check_lax_distributive_law(self, state: MotivationalState, stimulus: Stimulus, candidates: List[Action]) -> bool:
        """
        Validates the First Principle: Modular Appraisal-Decision Interface[cite: 287, 322].
        Checks that \lambda_X : \Psi(\mathbb{D}(X)) \Rightarrow \mathbb{D}(\Psi(X)) commutes up to a controlled error[cite: 308].
        """
        # Path 1: Appraise then Decide -> \mathbb{D}(\Psi(X))
        appraised_state_1 = self.appraisal.appraise(state, stimulus)
        action_1, final_state_1 = self.decision.decide(appraised_state_1, candidates)
        
        # Path 2: Decide then Appraise -> \Psi(\mathbb{D}(X))
        # Note: In a fully deployed system, the stimulus here would technically 
        # include the post-action environment context[cite: 318, 319].
        action_2, decided_state_2 = self.decision.decide(state, candidates)
        final_state_2 = self.appraisal.appraise(decided_state_2, stimulus)
        
        # Calculate the controlled distortion distance[cite: 332, 344].
        distortion = final_state_1.distance_to(final_state_2)
        
        # The law holds if the distortion is bounded by the acceptable delta[cite: 344].
        return distortion <= LAX_DISTRIBUTIVE_DELTA