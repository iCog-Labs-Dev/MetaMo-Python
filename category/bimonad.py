from typing import List, Tuple
import numpy as np
# Assuming these are available in your python path
from core.state import MotivationalState, Stimulus, Action
from core.config import (
    G_ETHIC,
    LAX_DISTRIBUTIVE_DELTA,
    G_IND,
    G_HELP,
    G_NOVEL,
    G_SELF,
    G_SOC,
    G_TRANS,
    G_CURIO,
    M_APPROACH,
    M_AROUSAL,
    M_RESOLUTION,
    M_SECURING,
    M_THRESHOLD,
    M_VALENCE,
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

    def _state_from_delta(self, decision_state: MotivationalState, proposed_delta_g: np.ndarray) -> MotivationalState:
        """
        Apply a proposed goal update inside the same stabilization path used by the main transition.
        """
        damped_delta_g = apply_homeostatic_damping(decision_state, proposed_delta_g)
        next_state = MotivationalState(
            G=np.clip(decision_state.G + damped_delta_g, 0.0, 1.0),
            M=decision_state.M.copy(),
        )
        return project_to_safe_region(next_state)

    def _decision_context(self, state: MotivationalState, stimulus: Stimulus) -> MotivationalState:
        """
        Build the post-appraisal state that the decision monad should score.
        """
        appraised_state = self.appraisal.appraise(state, stimulus)
        return raise_boundary_caution(appraised_state)

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

    def consensus_action(
        self,
        state_a: MotivationalState,
        state_b: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
    ) -> Action:
        """
        Select a shared action by combining the two subsystem evaluations over the same candidate set.
        """
        if not hasattr(self.decision, "score_candidate"):
            raise TypeError("decision monad must provide score_candidate for consensus action selection")

        context_a = self._decision_context(state_a, stimulus)
        context_b = self._decision_context(state_b, stimulus)

        best_action = None
        best_score = -float("inf")

        for candidate in candidates:
            score_a = self.decision.score_candidate(context_a, candidate)
            score_b = self.decision.score_candidate(context_b, candidate)
            mean_score = (score_a + score_b) / 2.0
            disagreement_penalty = 0.25 * abs(score_a - score_b)
            consensus_score = mean_score - disagreement_penalty

            if consensus_score > best_score:
                best_score = consensus_score
                best_action = candidate

        return best_action

    def consensus_transition(
        self,
        state_a: MotivationalState,
        state_b: MotivationalState,
        stimulus: Stimulus,
        candidates: List[Action],
    ) -> Tuple[Action, MotivationalState]:
        """
        Build a coupled consensus action and consensus target state from the same shared candidate set.
        """
        action = self.consensus_action(state_a, state_b, stimulus, candidates)
        context_a = self._decision_context(state_a, stimulus)
        context_b = self._decision_context(state_b, stimulus)
        target_a = self._state_from_delta(context_a, action.delta_g)
        target_b = self._state_from_delta(context_b, action.delta_g)
        merged_target = self.parallel_merge(target_a, target_b)
        return action, merged_target

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
        # Path 1: Appraise then Decide -> stabilized D(Psi(X))
        decision_state_1 = self._decision_context(state, stimulus)
        action_1, delta_g_1 = self.decision.decide(decision_state_1, candidates)
        final_state_1 = self._state_from_delta(decision_state_1, delta_g_1)
        
        # Path 2: Decide then Appraise -> stabilized Psi(D(X))
        action_2, delta_g_2 = self.decision.decide(state, candidates)
        decided_state_2 = self._state_from_delta(state, delta_g_2)
        final_state_2 = self._decision_context(decided_state_2, stimulus)
        
        # Calculate the controlled distortion distance[cite: 332, 344].
        distortion = final_state_1.distance_to(final_state_2)
        
        # The law holds if the distortion is bounded by the acceptable delta[cite: 344].
        return distortion <= LAX_DISTRIBUTIVE_DELTA
    
    def parallel_merge(self, state_a: MotivationalState, state_b: MotivationalState, coherence_correction: float = 0.05) -> MotivationalState:
        """
        Implements Principle 3: Parallel Motivational Compositionality.
        Witnesses the lax-monoidal structure \phi_{X,Y} of the composite F.
        Merges two parallel motivational subsystems with dimension-wise coherence corrections.
        Safety-relevant disagreements are merged conservatively, while exploratory disagreements
        are damped unless both subsystems support them.
        """
        weight_a = state_a.G[G_IND]
        weight_b = state_b.G[G_IND]
        total_weight = weight_a + weight_b + 1e-9

        base_G = ((state_a.G * weight_a) + (state_b.G * weight_b)) / total_weight
        base_M = ((state_a.M * weight_a) + (state_b.M * weight_b)) / total_weight

        disagreement_G = np.abs(state_a.G - state_b.G)
        disagreement_M = np.abs(state_a.M - state_b.M)

        consensus_G = base_G.copy()
        consensus_M = base_M.copy()

        # Safety-critical dimensions preserve the stronger caution/ethics signal under disagreement.
        safety_goal_idx = np.array([G_IND, G_HELP, G_ETHIC])
        consensus_G[safety_goal_idx] = np.maximum(state_a.G[safety_goal_idx], state_b.G[safety_goal_idx])

        # Exploratory dimensions require stronger agreement; otherwise they are damped toward the shared floor.
        exploratory_goal_idx = np.array([G_TRANS, G_CURIO, G_NOVEL, G_SELF])
        consensus_G[exploratory_goal_idx] = np.minimum(state_a.G[exploratory_goal_idx], state_b.G[exploratory_goal_idx])

        # Social engagement is shared but should not outrun subsystem agreement.
        consensus_G[G_SOC] = min(base_G[G_SOC], state_a.G[G_SOC], state_b.G[G_SOC])

        # Caution modulators preserve the higher warning signal.
        caution_mod_idx = np.array([M_THRESHOLD, M_SECURING])
        consensus_M[caution_mod_idx] = np.maximum(state_a.M[caution_mod_idx], state_b.M[caution_mod_idx])

        # Exploratory modulators are damped unless both subsystems align.
        exploratory_mod_idx = np.array([M_AROUSAL, M_APPROACH])
        consensus_M[exploratory_mod_idx] = np.minimum(state_a.M[exploratory_mod_idx], state_b.M[exploratory_mod_idx])

        # Valence/resolution remain closer to the weighted consensus.
        shared_mod_idx = np.array([M_VALENCE, M_RESOLUTION])
        consensus_M[shared_mod_idx] = (
            (state_a.M[shared_mod_idx] + state_b.M[shared_mod_idx]) / 2.0
        )

        goal_correction_scale = np.ones_like(base_G)
        goal_correction_scale[safety_goal_idx] = 1.5
        goal_correction_scale[exploratory_goal_idx] = 1.0
        goal_correction_scale[G_SOC] = 0.8

        mod_correction_scale = np.ones_like(base_M)
        mod_correction_scale[caution_mod_idx] = 1.5
        mod_correction_scale[exploratory_mod_idx] = 1.0
        mod_correction_scale[shared_mod_idx] = 0.8

        goal_correction = np.clip(coherence_correction * disagreement_G * goal_correction_scale, 0.0, 1.0)
        mod_correction = np.clip(coherence_correction * disagreement_M * mod_correction_scale, 0.0, 1.0)

        merged_G = base_G + goal_correction * (consensus_G - base_G)
        merged_M = base_M + mod_correction * (consensus_M - base_M)

        return MotivationalState(G=merged_G, M=merged_M)
