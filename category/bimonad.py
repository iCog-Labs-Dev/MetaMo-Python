from typing import List, Tuple
import numpy as np
# Assuming these are available in your python path
from core.state import MotivationalState, Stimulus, Action
from core.config import (
    LAX_DISTRIBUTIVE_DELTA,
    G_IND

    )
from category.functors import AppraisalComonad, DecisionMonad
from dynamics.stability import (
    apply_homeostatic_damping,
    check_contractive_update_law,
    is_in_safe_region,
    project_to_safe_region,
    raise_boundary_caution,
)

class MetaMoPseudoBimonad:
    """
    Represents the composite appraisal-then-decision operator F = D \circ \Psi.
    This forms a pseudo-bimonad on the motivational state space X = G \times M[cite: 28, 309].
    """
    def __init__(self, appraisal: AppraisalComonad, decision: DecisionMonad):
        self.appraisal = appraisal
        self.decision = decision

    def _compute_transition(self, state: MotivationalState, stimulus: Stimulus, candidates: List[Action]) -> Tuple[Action, MotivationalState]:
        """
        Compute one appraisal/decision transition before runtime validation.
        """
        # 1. Appraise (\Psi) - Update modulators based on stimulus[cite: 314].
        appraised_state = self.appraisal.appraise(state, stimulus)
        appraised_state = raise_boundary_caution(appraised_state)
        
        # 2. Decide (\mathbb{D}) - Score candidates and update goals[cite: 315].
        chosen_action, proposed_delta_g = self.decision.decide(appraised_state, candidates)

        damped_delta_g = apply_homeostatic_damping(appraised_state, proposed_delta_g)
        next_state = MotivationalState(
            G=np.clip(appraised_state.G + damped_delta_g, 0.0, 1.0),
            M=appraised_state.M.copy(),
        )
        next_state = project_to_safe_region(next_state)
        
        return chosen_action, next_state

    def _local_reference_state(self, state: MotivationalState, next_state: MotivationalState) -> MotivationalState:
        """
        Build a nearby state to probe local contractivity without depending on another subsystem.
        """
        delta_G = next_state.G - state.G
        delta_M = next_state.M - state.M

        probe_G = np.where(np.abs(delta_G) > 1e-6, np.sign(delta_G) * 0.01, 0.01)
        probe_M = np.where(np.abs(delta_M) > 1e-6, np.sign(delta_M) * 0.01, 0.01)

        return MotivationalState(
            G=np.clip(state.G + probe_G, 0.0, 1.0),
            M=np.clip(state.M + probe_M, 0.0, 1.0),
        )

    def _apply_conservative_fallback(self, current_state: MotivationalState, next_state: MotivationalState) -> MotivationalState:
        """
        Shrink the transition toward the current state when runtime checks fail.
        """
        fallback_state = MotivationalState(
            G=((current_state.G * 0.5) + (next_state.G * 0.5)),
            M=((current_state.M * 0.5) + (next_state.M * 0.5)),
        )
        return project_to_safe_region(fallback_state)

    def step(
        self,
        state: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
    ) -> Tuple[Action, MotivationalState]:
        """
        Executes one full cycle of F = D \circ \Psi.
        This governs the motivational coalgebra \alpha: X \to F(X)[cite: 310, 316].
        """
        chosen_action, next_state = self._compute_transition(state, stimulus, candidates)
        reference_state = self._local_reference_state(state, next_state)
        if not self.check_lax_distributive_law(state, stimulus, candidates):
            next_state = self._apply_conservative_fallback(state, next_state)
        if not check_contractive_update_law(self, state, reference_state, stimulus, candidates):
            next_state = self._apply_conservative_fallback(state, next_state)
        if not is_in_safe_region(next_state):
            next_state = self._apply_conservative_fallback(state, next_state)
        return chosen_action, next_state

    def check_lax_distributive_law(self, state: MotivationalState, stimulus: Stimulus, candidates: List[Action]) -> bool:
        """
        Validates the First Principle: Modular Appraisal-Decision Interface[cite: 287, 322].
        Checks that \lambda_X : \Psi(\mathbb{D}(X)) \Rightarrow \mathbb{D}(\Psi(X)) commutes up to a controlled error[cite: 308].
        """
        # Path 1: Appraise then Decide -> \mathbb{D}(\Psi(X))
        appraised_state_1 = self.appraisal.appraise(state, stimulus)
        action_1, delta_g_1 = self.decision.decide(appraised_state_1, candidates)
        final_state_1 = MotivationalState(
            G=np.clip(appraised_state_1.G + delta_g_1, 0.0, 1.0),
            M=appraised_state_1.M.copy(),
        )
        
        # Path 2: Decide then Appraise -> \Psi(\mathbb{D}(X))
        # Note: In a fully deployed system, the stimulus here would technically 
        # include the post-action environment context[cite: 318, 319].
        action_2, delta_g_2 = self.decision.decide(state, candidates)
        decided_state_2 = MotivationalState(
            G=np.clip(state.G + delta_g_2, 0.0, 1.0),
            M=state.M.copy(),
        )
        final_state_2 = self.appraisal.appraise(decided_state_2, stimulus)
        
        # Calculate the controlled distortion distance[cite: 332, 344].
        distortion = final_state_1.distance_to(final_state_2)
        
        # The law holds if the distortion is bounded by the acceptable delta[cite: 344].
        return distortion <= LAX_DISTRIBUTIVE_DELTA
    def parallel_merge(self, state_a: MotivationalState, state_b: MotivationalState, coherence_correction: float = 0.05) -> MotivationalState:
        """
        Implements Principle 3: Parallel Motivational Compositionality.
        Witnesses the lax-monoidal structure \phi_{X,Y} of the composite F.
        Merges two parallel motivational subsystems with small coherence corrections.
        """
        # Merge goals using a weighted average based on their respective Individuation drives
        weight_a = state_a.G[G_IND]
        weight_b = state_b.G[G_IND]
        total_weight = weight_a + weight_b + 1e-9 # Prevent division by zero
        
        merged_G = ((state_a.G * weight_a) + (state_b.G * weight_b)) / total_weight
        merged_M = ((state_a.M * weight_a) + (state_b.M * weight_b)) / total_weight
        
        # Apply the small topological coherence correction to prevent subsystem interference
        merged_G = np.clip(merged_G - coherence_correction, 0.0, 1.0)
        
        return MotivationalState(G=merged_G, M=merged_M)