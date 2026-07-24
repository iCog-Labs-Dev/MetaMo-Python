from typing import Any, List, Tuple
import numpy as np
from dataclasses import dataclass
from core.state import MotivationalState, Action
from core.config import (
    LAX_DISTRIBUTIVE_DELTA,
    PARALLEL_COMPOSITION_DELTA,
    )
from category.diagnostics import (
    MetaMoDiagnostics,
    MetaMoDiagnosticsHistory,
    MetaMoDiagnosticsSummary,
)
from category.functors import AppraisalComonad, DecisionMonad
from category.laws import StateLawCheckResult
from category.merge import DefaultParallelMergePolicy
from dynamics.coherence import DefaultCoherencePolicy
from dynamics.stability import (
    DefaultStabilityPolicy,
)


@dataclass(frozen=True)
class TransitionComputation:
    """
    Raw transition plus projection correction telemetry.
    """

    action: Action
    state: MotivationalState
    projection_delta: float


class MetaMoPseudoBimonad:
    """
    Represents the composite appraisal-then-decision operator F = D o ψ.
    This forms a pseudo-bimonad on the motivational state space X = G \times M.
    """
    def __init__(
        self,
        appraisal: AppraisalComonad,
        decision: DecisionMonad,
        stability_policy=None,
        coherence_policy=None,
        merge_policy=None,
    ):
        self.appraisal = appraisal
        self.decision = decision
        self.stability_policy = stability_policy or DefaultStabilityPolicy()
        self.coherence_policy = coherence_policy or DefaultCoherencePolicy()
        self.merge_policy = merge_policy or DefaultParallelMergePolicy()
        self.diagnostics_history = MetaMoDiagnosticsHistory()

    def _compute_transition_details(self, state: MotivationalState, stimulus: Any, candidates: List[Action]) -> TransitionComputation:
        """
        Compute one appraisal/decision transition before runtime validation.
        """
        # 1. Appraise - Update modulators based on stimulus.
        goal_change_feedback = self._goal_change_feedback(state, stimulus)
        appraised_state = self.appraisal.appraise(state, stimulus)
        appraised_state = self.stability_policy.raise_boundary_caution(appraised_state)

        # 2. Decide - Score candidates and update goals.
        chosen_action, proposed_delta_g = self.decision.decide(
            appraised_state,
            candidates,
            feedback=goal_change_feedback,
        )

        damped_delta_g = self.stability_policy.apply_homeostatic_damping(appraised_state, proposed_delta_g)
        next_state = MotivationalState(
            G=np.clip(appraised_state.G + damped_delta_g, 0.0, 1.0),
            M=appraised_state.M.copy(),
            schema=appraised_state.schema,
        )
        projected_state = self.stability_policy.project_to_safe_region(next_state)

        return TransitionComputation(
            action=chosen_action,
            state=projected_state,
            projection_delta=next_state.distance_to(projected_state),
        )

    def _compute_transition(self, state: MotivationalState, stimulus: Any, candidates: List[Action]) -> Tuple[Action, MotivationalState]:
        """
        Compute one appraisal/decision transition before runtime validation.
        """
        computation = self._compute_transition_details(state, stimulus, candidates)
        return computation.action, computation.state

    def _target_transition_details(
        self,
        state: MotivationalState,
        stimulus: Any,
        candidates: List[Action],
    ) -> TransitionComputation:
        computation = self._compute_transition_details(state, stimulus, candidates)
        next_state = computation.state
        projection_delta = computation.projection_delta
        reference_state = self._local_reference_state(state, next_state)

        if not self.check_lax_distributive_law(state, stimulus, candidates):
            fallback_state = self._apply_conservative_fallback(state, next_state)
            projection_delta += next_state.distance_to(fallback_state)
            next_state = fallback_state

        if not self.stability_policy.check_contractive_update_law(self, state, reference_state, stimulus, candidates):
            fallback_state = self._apply_conservative_fallback(state, next_state)
            projection_delta += next_state.distance_to(fallback_state)
            next_state = fallback_state

        if not self.stability_policy.is_in_safe_region(next_state):
            fallback_state = self._apply_conservative_fallback(state, next_state)
            projection_delta += next_state.distance_to(fallback_state)
            next_state = fallback_state

        return TransitionComputation(
            action=computation.action,
            state=next_state,
            projection_delta=projection_delta,
        )

    def target_transition(
        self,
        state: MotivationalState,
        stimulus: Any,
        candidates: List[Action],
    ) -> Tuple[Action, MotivationalState]:
        """
        Compute a stabilized target state before incremental embodiment.
        """
        computation = self._target_transition_details(state, stimulus, candidates)
        return computation.action, computation.state

    def _state_from_delta(self, decision_state: MotivationalState, proposed_delta_g: np.ndarray) -> MotivationalState:
        """
        Apply a proposed goal update inside the same stabilization path used by the main transition.
        """
        damped_delta_g = self.stability_policy.apply_homeostatic_damping(decision_state, proposed_delta_g)
        next_state = MotivationalState(
            G=np.clip(decision_state.G + damped_delta_g, 0.0, 1.0),
            M=decision_state.M.copy(),
            schema=decision_state.schema,
        )
        return self.stability_policy.project_to_safe_region(next_state)

    def _decision_context(self, state: MotivationalState, stimulus: Any) -> MotivationalState:
        """
        Build the post-appraisal state that the decision monad should score.
        """
        appraised_state = self.appraisal.appraise(state, stimulus)
        return self.stability_policy.raise_boundary_caution(appraised_state)

    def _goal_change_feedback(self, state: MotivationalState, stimulus: Any) -> Any:
        if hasattr(self.appraisal, "goal_change_feedback"):
            return self.appraisal.goal_change_feedback(state, stimulus)
        return None

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
            schema=state.schema,
        )

    def consensus_action(
        self,
        state_a: MotivationalState,
        state_b: MotivationalState,
        stimulus: Any,
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
        stimulus: Any,
        candidates: List[Action],
    ) -> Tuple[Action, MotivationalState]:
        """
        Build a coupled consensus action and consensus target state from the same shared candidate set.
        """
        action = self.consensus_action(state_a, state_b, stimulus, candidates)
        context_a = self._decision_context(state_a, stimulus)
        context_b = self._decision_context(state_b, stimulus)
        feedback_a = self._goal_change_feedback(state_a, stimulus)
        feedback_b = self._goal_change_feedback(state_b, stimulus)
        delta_a = self._proposed_delta_for_action(context_a, action, feedback_a)
        delta_b = self._proposed_delta_for_action(context_b, action, feedback_b)
        target_a = self._state_from_delta(context_a, delta_a)
        target_b = self._state_from_delta(context_b, delta_b)
        merged_target = self.parallel_merge(target_a, target_b)
        return action, merged_target

    def _proposed_delta_for_action(
        self,
        state: MotivationalState,
        action: Action,
        feedback: Any = None,
    ) -> np.ndarray:
        """
        Use profile-derived MAGUS Delta G when the decision layer exposes it.
        """
        if hasattr(self.decision, "propose_delta_g"):
            return self.decision.propose_delta_g(state, action, feedback)
        return action.delta_g.copy()

    def _apply_conservative_fallback(self, current_state: MotivationalState, next_state: MotivationalState) -> MotivationalState:
        """
        Shrink the transition toward the current state when runtime checks fail.
        """
        fallback_state = MotivationalState(
            G=((current_state.G * 0.5) + (next_state.G * 0.5)),
            M=((current_state.M * 0.5) + (next_state.M * 0.5)),
            schema=current_state.schema,
        )
        return self.stability_policy.project_to_safe_region(fallback_state)

    def step(
        self,
        state: MotivationalState,
        stimulus: Any,
        candidates: List[Action],
        embody: bool = True,
        record_diagnostics: bool = True,
    ) -> Tuple[Action, MotivationalState]:
        """
        Executes one full cycle of F = D(ψ(X)).

        By default this returns the incrementally embodied next state required
        by Principle 5. Set embody=False to inspect the stabilized target.
        """
        chosen_action, next_state, _ = self.step_with_diagnostics(
            state,
            stimulus,
            candidates,
            embody=embody,
            record_diagnostics=record_diagnostics,
        )
        return chosen_action, next_state

    def step_with_diagnostics(
        self,
        state: MotivationalState,
        stimulus: Any,
        candidates: List[Action],
        embody: bool = True,
        record_diagnostics: bool = True,
    ) -> Tuple[Action, MotivationalState, MetaMoDiagnostics]:
        """
        Executes one full cycle and returns telemetry for the principle checks.
        """
        lax_result = self.measure_lax_distributive_law(state, stimulus, candidates)
        target_computation = self._target_transition_details(state, stimulus, candidates)
        chosen_action = target_computation.action
        target_state = target_computation.state
        reference_state = self._local_reference_state(state, target_state)
        contractive_holds = self.stability_policy.check_contractive_update_law(
            self,
            state,
            reference_state,
            stimulus,
            candidates,
        )

        if embody:
            blend_result = self.coherence_policy.measure_blend(state, target_state)
            next_state = blend_result.state
            drift = blend_result.drift
            blend_alpha = blend_result.alpha
            base_blend_alpha = blend_result.base_alpha
        else:
            next_state = target_state
            drift = self.coherence_policy.measure_self_model_drift(state, next_state)
            blend_alpha = 1.0
            base_blend_alpha = 1.0

        diagnostics = MetaMoDiagnostics(
            action_id=chosen_action.id,
            lax_error=lax_result.error,
            lax_tolerance=lax_result.tolerance,
            lax_holds=lax_result.holds,
            contractive_holds=contractive_holds,
            target_in_safe_region=self.stability_policy.is_in_safe_region(target_state),
            final_in_safe_region=self.stability_policy.is_in_safe_region(next_state),
            boundary_pressure_before=self.stability_policy.boundary_pressure(state),
            boundary_pressure_target=self.stability_policy.boundary_pressure(target_state),
            boundary_pressure_final=self.stability_policy.boundary_pressure(next_state),
            projection_delta=target_computation.projection_delta,
            target_distance=state.distance_to(target_state),
            state_drift=drift.state_distance,
            self_model_drift=drift.self_model_distance,
            combined_self_model_drift=drift.combined_drift,
            self_model_drift_tolerance=drift.max_allowed_drift,
            self_model_drift_holds=drift.holds,
            blend_alpha=blend_alpha,
            base_blend_alpha=base_blend_alpha,
        )

        if record_diagnostics:
            self.diagnostics_history.append(diagnostics)
        return chosen_action, next_state, diagnostics

    def measure_lax_distributive_law(
        self,
        state: MotivationalState,
        stimulus: Any,
        candidates: List[Action],
        tolerance: float = LAX_DISTRIBUTIVE_DELTA,
    ) -> StateLawCheckResult:
        """
        Measures the First Principle: Modular Appraisal-Decision Interface.
        """
        goal_change_feedback = self._goal_change_feedback(state, stimulus)

        # Path 1: Appraise then Decide -> stabilized D(Psi(X))
        decision_state_1 = self._decision_context(state, stimulus)
        action_1, delta_g_1 = self.decision.decide(
            decision_state_1,
            candidates,
            feedback=goal_change_feedback,
        )
        final_state_1 = self._state_from_delta(decision_state_1, delta_g_1)

        # Path 2: Decide then Appraise -> stabilized Psi(D(X))
        action_2, delta_g_2 = self.decision.decide(
            state,
            candidates,
            feedback=goal_change_feedback,
        )
        decided_state_2 = self._state_from_delta(state, delta_g_2)
        final_state_2 = self._decision_context(decided_state_2, stimulus)

        # Calculate the controlled distortion distance.
        distortion = final_state_1.distance_to(final_state_2)

        return StateLawCheckResult(
            principle="modular_appraisal_decision_interface",
            left_state=final_state_1,
            right_state=final_state_2,
            error=distortion,
            tolerance=tolerance,
            holds=distortion <= tolerance,
        )

    def check_lax_distributive_law(self, state: MotivationalState, stimulus: Any, candidates: List[Action]) -> bool:
        """
        Validates the First Principle: Modular Appraisal-Decision Interface.
        """
        return self.measure_lax_distributive_law(state, stimulus, candidates).holds

    def measure_parallel_compositionality(
        self,
        state_a: MotivationalState,
        state_b: MotivationalState,
        stimulus: Any,
        candidates: List[Action],
        tolerance: float = PARALLEL_COMPOSITION_DELTA,
    ) -> StateLawCheckResult:
        """
        Measures Principle 3 by comparing merge-after-update with update-after-merge.
        """
        _, next_a = self._compute_transition(state_a, stimulus, candidates)
        _, next_b = self._compute_transition(state_b, stimulus, candidates)
        left_state = self.parallel_merge(next_a, next_b)

        merged_state = self.parallel_merge(state_a, state_b)
        _, right_state = self._compute_transition(merged_state, stimulus, candidates)

        error = left_state.distance_to(right_state)
        return StateLawCheckResult(
            principle="parallel_motivational_compositionality",
            left_state=left_state,
            right_state=right_state,
            error=error,
            tolerance=tolerance,
            holds=error <= tolerance,
        )

    def check_parallel_compositionality(
        self,
        state_a: MotivationalState,
        state_b: MotivationalState,
        stimulus: Any,
        candidates: List[Action],
    ) -> bool:
        """
        Validates Principle 3 with the configured coherence tolerance.
        """
        return self.measure_parallel_compositionality(state_a, state_b, stimulus, candidates).holds

    def parallel_merge(self, state_a: MotivationalState, state_b: MotivationalState, coherence_correction: float = 0.05) -> MotivationalState:
        """
        Implements Principle 3: Parallel Motivational Compositionality.
        Witnesses the lax-monoidal structure.
        Merges two parallel motivational subsystems with schema-aware coherence corrections.
        """
        return self.merge_policy.merge(
            state_a,
            state_b,
            decision=self.decision,
            coherence_correction=coherence_correction,
        )
