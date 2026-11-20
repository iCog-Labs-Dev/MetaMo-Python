from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from applications.gridworld.profile import (
    GRIDWORLD_APPRAISAL_PROFILE,
    GRIDWORLD_DECISION_PROFILE,
    GRIDWORLD_PROFILE,
)
from category.bimonad import MetaMoPseudoBimonad
from category.diagnostics import MetaMoDiagnostics
from core.state import Action, MotivationalState
from dynamics.coherence import blend_states
from dynamics.stability import is_in_safe_region, project_to_safe_region
from magus.decision import MagusDecision
from openpsi.appraisal import OpenPsiAppraisal


@dataclass(frozen=True)
class GridWorldRuntime:
    """Bound application profile and bimonad for the GridWorld use case."""

    name: str
    profile: Any
    appraisal_profile: Any
    decision_profile: Any
    bimonad: MetaMoPseudoBimonad


@dataclass(frozen=True)
class CandidatePerspectiveScores:
    """Action scores produced by the parallel task and safety perspectives."""

    task: np.ndarray
    safety: np.ndarray
    consensus: np.ndarray
    disagreement: np.ndarray


@dataclass(frozen=True)
class GridWorldCompositionalityAudit:
    """Application-level evidence for Principle 3 on one decision state."""

    error: float
    tolerance: float
    holds: bool
    max_goal_error: float
    max_goal_coordinate: str
    max_modulator_error: float
    max_modulator_coordinate: str
    parallel_action_id: str
    merged_first_action_id: str
    action_holds: bool


def make_gridworld_runtime() -> GridWorldRuntime:
    """Create an isolated rule-appraisal GridWorld runtime."""
    appraisal = OpenPsiAppraisal(profile=GRIDWORLD_APPRAISAL_PROFILE)
    return GridWorldRuntime(
        name="gridworld:rule",
        profile=GRIDWORLD_PROFILE,
        appraisal_profile=appraisal.profile,
        decision_profile=GRIDWORLD_DECISION_PROFILE,
        bimonad=MetaMoPseudoBimonad(
            appraisal,
            MagusDecision(profile=GRIDWORLD_DECISION_PROFILE),
        ),
    )


RUNTIME = make_gridworld_runtime()


def get_runtime() -> GridWorldRuntime:
    """Return the single GridWorld MetaMo runtime."""
    return RUNTIME


def safety_threshold(mot_state: MotivationalState) -> float:
    """Safety threshold proxy for the gridworld dashboard."""
    return RUNTIME.profile.safety_threshold(mot_state)


def arousal(mot_state: MotivationalState) -> float:
    """Arousal proxy for the gridworld dashboard."""
    return RUNTIME.profile.arousal(mot_state)


def in_safe_region(mot_state: MotivationalState) -> bool:
    """Checks whether a motivational state is inside the MetaMo safe region R."""
    return is_in_safe_region(mot_state)


def build_stimulus(
    env_state: dict,
    mot_state: Optional[MotivationalState] = None,
    runtime: Optional[GridWorldRuntime] = None,
) -> Any:
    """Build the MetaMo appraisal stimulus from the current environment state."""
    selected_runtime = runtime or RUNTIME
    return selected_runtime.profile.build_stimulus(env_state, mot_state)


def build_candidates(
    env_state: dict,
    mot_state: Optional[MotivationalState] = None,
    runtime: Optional[GridWorldRuntime] = None,
) -> List[Action]:
    """Generate motivational action candidates for all movement directions."""
    selected_runtime = runtime or RUNTIME
    return selected_runtime.profile.build_candidates(env_state, mot_state)


def build_consensus_states(
    env_state: dict,
    mot_state: MotivationalState,
    stimulus: Any,
    runtime: Optional[GridWorldRuntime] = None,
) -> tuple[MotivationalState, MotivationalState]:
    """Build the safety and growth perspectives used for consensus transition."""
    selected_runtime = runtime or RUNTIME
    return selected_runtime.profile.build_consensus_states(env_state, mot_state, stimulus)


def perspective_candidate_scores(
    mot_state: MotivationalState,
    stimulus: Any,
    candidates: List[Action],
    env_state: Optional[dict] = None,
    runtime: Optional[GridWorldRuntime] = None,
) -> CandidatePerspectiveScores:
    """Return explicit task, safety, and legacy consensus scores per action.

    The task branch uses the growth perspective and only the task-grounded
    primary goals.  The safety branch uses the safety perspective, energy
    preservation, individuation, and anti-goal penalties.  ``consensus``
    preserves the former full MAGUS consensus score for legacy ablations.
    """
    selected_runtime = runtime or RUNTIME
    if env_state is None:
        context_a = selected_runtime.bimonad._decision_context(mot_state, stimulus)
        context_b = context_a
    else:
        safety_state, growth_state = build_consensus_states(
            env_state,
            mot_state,
            stimulus,
            runtime=selected_runtime,
        )
        context_a = selected_runtime.bimonad._decision_context(safety_state, stimulus)
        context_b = selected_runtime.bimonad._decision_context(growth_state, stimulus)

    task_scores = []
    safety_scores = []
    consensus_scores = []
    disagreements = []
    for candidate in candidates:
        _, safety_breakdown = selected_runtime.bimonad.decision.evaluate_candidate(
            context_a,
            candidate,
        )
        _, task_breakdown = selected_runtime.bimonad.decision.evaluate_candidate(
            context_b,
            candidate,
        )

        # The task branch intentionally excludes risk, safe-exit, energy, and
        # anti-goal terms. This makes its action semantics independent from the
        # safety branch while retaining motivational goal/modulator weights.
        profile = selected_runtime.decision_profile
        progress_direction = float(
            np.clip(
                2.0 * float(candidate.metadata.get("mineral_progress", 0.5)) - 1.0,
                -1.0,
                1.0,
            )
        )
        collected = float(candidate.metadata.get("mineral_collected", 0.0))
        boundary_hit = float(candidate.metadata.get("boundary_hit", 0.0))
        mineral_idx = context_b.schema.goal_index("mineral_acquisition")
        navigation_idx = context_b.schema.goal_index("navigation_efficiency")
        mineral_weight = (
            context_b.G[mineral_idx]
            * profile.relevant_modulator(context_b, mineral_idx)
            * profile.compatibility_factor(mineral_idx, context_b, candidate)
        )
        navigation_weight = (
            context_b.G[navigation_idx]
            * profile.relevant_modulator(context_b, navigation_idx)
            * profile.compatibility_factor(navigation_idx, context_b, candidate)
        )
        mineral_alignment = float(
            np.clip(0.70 * progress_direction + 0.30 * collected, -1.0, 1.0)
        )
        navigation_alignment = float(
            np.clip(0.80 * progress_direction - 0.50 * boundary_hit, -1.0, 1.0)
        )
        task_score = (
            mineral_weight * mineral_alignment
            + navigation_weight * navigation_alignment
        )

        safety_primary = safety_breakdown.primary_goal_contributions
        safety_score = (
            safety_primary.get("energy_preservation", 0.0)
            - sum(safety_breakdown.anti_goal_penalties.values())
            + safety_breakdown.individuation_adjustment
            + 0.25
            * context_a.goal("individuation")
            * float(candidate.metadata.get("safe_exit_delta", 0.0))
        )

        score_a = safety_breakdown.total_score
        score_b = task_breakdown.total_score
        disagreement = abs(score_a - score_b)
        consensus_score = ((score_a + score_b) / 2.0) - (0.25 * disagreement)

        task_scores.append(task_score)
        safety_scores.append(safety_score)
        consensus_scores.append(consensus_score)
        disagreements.append(abs(task_score - safety_score))

    return CandidatePerspectiveScores(
        task=np.asarray(task_scores, dtype=float),
        safety=np.asarray(safety_scores, dtype=float),
        consensus=np.asarray(consensus_scores, dtype=float),
        disagreement=np.asarray(disagreements, dtype=float),
    )


def measure_gridworld_compositionality(
    env_state: dict,
    mot_state: MotivationalState,
    stimulus: Any,
    candidates: List[Action],
    runtime: Optional[GridWorldRuntime] = None,
) -> GridWorldCompositionalityAudit:
    """Audit merge-after-update versus update-after-merge for GridWorld."""
    selected_runtime = runtime or RUNTIME
    safety_state, growth_state = build_consensus_states(
        env_state,
        mot_state,
        stimulus,
        runtime=selected_runtime,
    )
    law_result = selected_runtime.bimonad.measure_parallel_compositionality(
        safety_state,
        growth_state,
        stimulus,
        candidates,
    )
    goal_errors = np.abs(law_result.left_state.G - law_result.right_state.G)
    modulator_errors = np.abs(law_result.left_state.M - law_result.right_state.M)
    goal_idx = int(np.argmax(goal_errors))
    modulator_idx = int(np.argmax(modulator_errors))

    parallel_action = selected_runtime.bimonad.consensus_action(
        safety_state,
        growth_state,
        stimulus,
        candidates,
    )
    merged_state = selected_runtime.bimonad.parallel_merge(safety_state, growth_state)
    merged_first_action, _ = selected_runtime.bimonad._compute_transition(
        merged_state,
        stimulus,
        candidates,
    )
    return GridWorldCompositionalityAudit(
        error=float(law_result.error),
        tolerance=float(law_result.tolerance),
        holds=bool(law_result.holds),
        max_goal_error=float(goal_errors[goal_idx]),
        max_goal_coordinate=mot_state.schema.goal_names[goal_idx],
        max_modulator_error=float(modulator_errors[modulator_idx]),
        max_modulator_coordinate=mot_state.schema.modulator_names[modulator_idx],
        parallel_action_id=parallel_action.id,
        merged_first_action_id=merged_first_action.id,
        action_holds=parallel_action.id == merged_first_action.id,
    )


def transition_for_action(
    env_state: dict,
    mot_state: MotivationalState,
    action_idx: int,
    stimulus: Optional[Any] = None,
    candidates: Optional[List[Action]] = None,
    runtime: Optional[GridWorldRuntime] = None,
) -> tuple[Action, MotivationalState, Any, MotivationalState]:
    """Apply a selected grid action through bimonad.consensus_transition."""
    action, next_state, stimulus, target_state, _ = transition_for_action_with_diagnostics(
        env_state,
        mot_state,
        action_idx,
        stimulus=stimulus,
        candidates=candidates,
        runtime=runtime,
    )
    return action, next_state, stimulus, target_state


def transition_for_action_with_diagnostics(
    env_state: dict,
    mot_state: MotivationalState,
    action_idx: int,
    stimulus: Optional[Any] = None,
    candidates: Optional[List[Action]] = None,
    runtime: Optional[GridWorldRuntime] = None,
) -> tuple[Action, MotivationalState, Any, MotivationalState, MetaMoDiagnostics]:
    """Apply a selected grid action and record MetaMo diagnostics."""
    selected_runtime = runtime or RUNTIME
    stimulus = stimulus or build_stimulus(env_state, mot_state, runtime=selected_runtime)
    candidates = candidates or build_candidates(env_state, mot_state, runtime=selected_runtime)
    selected = [candidates[action_idx]]
    safety_state, growth_state = build_consensus_states(
        env_state,
        mot_state,
        stimulus,
        runtime=selected_runtime,
    )
    action, target_state = selected_runtime.bimonad.consensus_transition(
        safety_state,
        growth_state,
        stimulus,
        selected,
    )
    target_state = project_to_safe_region(target_state)
    next_state = blend_states(mot_state, target_state)
    _, _, diagnostics = selected_runtime.bimonad.step_with_diagnostics(
        mot_state, stimulus, selected
    )
    return action, next_state, stimulus, target_state, diagnostics
