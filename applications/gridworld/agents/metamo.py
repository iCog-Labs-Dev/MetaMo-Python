"""MetaMo-enhanced GridWorld agent with a fair Q-learning substrate."""

import random
from typing import Optional

import numpy as np

from applications.gridworld.agents.q_features import encode_state, valid_actions
from applications.gridworld.agents.reward_shaping import RewardShapingConfig, learning_reward
from applications.gridworld.config import DEFAULT_EXTERNAL_RISK_WEIGHT
from applications.gridworld.runtime import (
    CandidatePerspectiveScores,
    GridWorldRuntime,
    build_candidates,
    build_stimulus,
    make_gridworld_runtime,
    measure_gridworld_compositionality,
    perspective_candidate_scores,
    transition_for_action,
)
from applications.gridworld.state import create_initial_motivational_state
from core.state import Action, MotivationalState


class MetaMoAgent:
    """Q-learning plus MetaMo scoring, with explicit experimental controls."""

    ACTIONS = 4

    def __init__(
        self,
        grid_size: int = 10,
        alpha: float = 0.3,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.97,
        seed: int = 42,
        motivation_weight: float = 6.0,
        risk_weight: float = DEFAULT_EXTERNAL_RISK_WEIGHT,
        exploration_bonus_weight: float = 0.0,
        appraisal_mode: str = "rule",
        max_distance_bin: int = 4,
        safe_exploration_probability: float = 0.7,
        progress_shaping_weight: float = 8.0,
        lava_shaping_penalty: float = 80.0,
        boundary_shaping_penalty: float = 5.0,
        danger_shaping_penalty: float = 2.0,
        mask_lava_on_exploit: bool = False,
        q_encoder: str = "compact_hazard",
        reward_mode: str = "shaped",
        allow_boundary_actions: bool = False,
        selector_mode: str = "composed",
        task_weight: float = 0.5,
        safety_weight: float = 0.5,
        disagreement_penalty: float = 0.25,
        q_absolute_tolerance: float = 2.0,
        q_relative_tolerance: float = 0.20,
        early_guidance_bonus: float = 2.0,
        hard_safety: bool = True,
        hard_safety_risk_threshold: float = 1.0,
        compositionality_audit_interval: int = 10,
    ):
        self.seed = seed
        self.grid_size = grid_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.motivation_weight = motivation_weight
        self.risk_weight = risk_weight
        self.exploration_bonus_weight = exploration_bonus_weight
        self.appraisal_mode = appraisal_mode
        self.max_distance_bin = max_distance_bin
        self.safe_exploration_probability = safe_exploration_probability
        self.mask_lava_on_exploit = mask_lava_on_exploit
        self.q_encoder = q_encoder
        self.reward_mode = reward_mode
        self.allow_boundary_actions = allow_boundary_actions
        if selector_mode not in {"legacy_additive", "task", "safety", "composed"}:
            raise ValueError(f"unknown selector mode: {selector_mode}")
        if task_weight < 0.0 or safety_weight < 0.0:
            raise ValueError("task and safety weights must be non-negative")
        if disagreement_penalty < 0.0:
            raise ValueError("disagreement penalty must be non-negative")
        if compositionality_audit_interval < 1:
            raise ValueError("compositionality audit interval must be positive")
        self.selector_mode = selector_mode
        self.task_weight = task_weight
        self.safety_weight = safety_weight
        self.disagreement_penalty = disagreement_penalty
        self.q_absolute_tolerance = q_absolute_tolerance
        self.q_relative_tolerance = q_relative_tolerance
        self.early_guidance_bonus = early_guidance_bonus
        self.hard_safety = hard_safety
        self.hard_safety_risk_threshold = hard_safety_risk_threshold
        self.compositionality_audit_interval = compositionality_audit_interval
        self.reward_shaping = RewardShapingConfig(
            progress_weight=progress_shaping_weight,
            lava_penalty=lava_shaping_penalty,
            boundary_penalty=boundary_shaping_penalty,
            danger_penalty=danger_shaping_penalty,
        )
        self.rng = random.Random(seed)

        self.runtime: GridWorldRuntime = make_gridworld_runtime(appraisal_mode)
        self.neutral_runtime: GridWorldRuntime = (
            self.runtime
            if appraisal_mode == "neutral"
            else make_gridworld_runtime("neutral")
        )
        self.mot = create_initial_motivational_state()
        self._pending_state: Optional[MotivationalState] = None
        self._pending_action: Optional[Action] = None
        self.q_table: dict[tuple, np.ndarray] = {}
        self.visit_counts: dict[tuple, np.ndarray] = {}

        self._decision_count = 0

    def reset_episode(self):
        """Reset motivational state and episode-local diagnostic logs."""
        self.mot = create_initial_motivational_state()
        self._pending_state = None
        self._pending_action = None
        self._decision_count = 0

    def _encode(self, state: dict) -> tuple:
        return encode_state(
            state,
            mode=self.q_encoder,
            grid_size=self.grid_size,
            max_distance_bin=self.max_distance_bin,
        )

    def _q_values(self, key: tuple) -> np.ndarray:
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.ACTIONS)
            self.visit_counts[key] = np.zeros(self.ACTIONS)
        return self.q_table[key]

    def _valid_actions(self, state: dict, avoid_lava: bool = False) -> list[int]:
        return valid_actions(
            state,
            grid_size=self.grid_size,
            avoid_lava=avoid_lava,
            allow_boundary=self.allow_boundary_actions,
        )

    def _select_exploratory_action(self, state: dict) -> int:
        avoid_lava = self.rng.random() < self.safe_exploration_probability
        return self.rng.choice(self._valid_actions(state, avoid_lava=avoid_lava))

    def _argmax_with_random_tie(self, scores: np.ndarray, actions: list[int]) -> int:
        max_score = max(scores[action] for action in actions)
        best = [action for action in actions if np.isclose(scores[action], max_score)]
        return self.rng.choice(best)

    @staticmethod
    def _diagnostic_argmax(scores: np.ndarray, actions: list[int]) -> int:
        """Deterministic argmax that does not perturb experimental RNG state."""
        return max(actions, key=lambda action: (float(scores[action]), -action))

    def _shape_reward(self, state: dict, reward: float, next_state: dict) -> float:
        return learning_reward(
            state,
            reward,
            next_state,
            mode=self.reward_mode,
            config=self.reward_shaping,
        )

    def select_q_action(
        self,
        state: dict,
        return_diagnostics: bool = False,
    ) -> int | tuple[int, dict]:
        """Select from Q alone using the baseline epsilon-greedy policy.

        This deliberately skips appraisal, motivational transition creation,
        composed scoring, and hard-safety filtering.  It lets the evaluator
        withdraw MetaMo without replacing the learned Q-table or changing the
        agent's learning configuration.
        """
        key = self._encode(state)
        q_values = self._q_values(key)
        visit_counts = self.visit_counts[key]
        actions = self._valid_actions(
            state,
            avoid_lava=self.mask_lava_on_exploit,
        )
        q_greedy_action = self._diagnostic_argmax(q_values, actions)
        exploratory = self.rng.random() < self.epsilon
        if exploratory:
            action_idx = self._select_exploratory_action(state)
        else:
            action_idx = self._argmax_with_random_tie(q_values, actions)

        if not return_diagnostics:
            return action_idx

        candidates = build_candidates(state, self.mot, runtime=self.runtime)
        risk_estimates = np.asarray(
            [
                float(
                    candidate.metadata.get(
                        "hazard_risk",
                        candidate.metadata.get("risk", 0.0),
                    )
                )
                for candidate in candidates
            ],
            dtype=float,
        )
        threshold = self.hard_safety_risk_threshold
        return action_idx, {
            "exploratory": exploratory,
            "q_greedy_action": q_greedy_action,
            "executed_action": action_idx,
            "q_proposed_hazard_risk": float(risk_estimates[q_greedy_action]),
            "executed_hazard_risk": float(risk_estimates[action_idx]),
            "q_proposed_immediate_lava": bool(
                risk_estimates[q_greedy_action] >= threshold
            ),
            "executed_immediate_lava": bool(risk_estimates[action_idx] >= threshold),
            "q_proposal_blocked": False,
            "q_proposed_visit_count": float(visit_counts[q_greedy_action]),
            "q_proposed_value": float(q_values[q_greedy_action]),
            "executed_changed_q_action": action_idx != q_greedy_action,
        }

    @staticmethod
    def _normalize_preferences(scores: np.ndarray, actions: list[int]) -> np.ndarray:
        """Center and bound an action preference vector over eligible actions."""
        normalized = np.zeros_like(scores, dtype=float)
        if not actions:
            return normalized
        values = np.asarray([scores[action] for action in actions], dtype=float)
        centered = values - float(np.mean(values))
        scale = float(np.max(np.abs(centered)))
        if scale > 1e-12:
            centered = centered / scale
        for action, value in zip(actions, centered):
            normalized[action] = float(value)
        return normalized

    def _composed_preferences(
        self,
        scores: CandidatePerspectiveScores,
        actions: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        task = self._normalize_preferences(scores.task, actions)
        safety = self._normalize_preferences(scores.safety, actions)
        disagreement = np.abs(task - safety)
        if self.selector_mode == "task":
            composed = task.copy()
        elif self.selector_mode == "safety":
            composed = safety.copy()
        else:
            composed = (
                self.task_weight * task
                + self.safety_weight * safety
                - self.disagreement_penalty * disagreement
            )
        return task, safety, disagreement, composed

    def _hard_safety_actions(
        self,
        actions: list[int],
        risk_estimates: np.ndarray,
    ) -> list[int]:
        """Reject immediate lava entry when a safer legal action exists."""
        if not self.hard_safety:
            return list(actions)
        safe = [
            action
            for action in actions
            if risk_estimates[action] < self.hard_safety_risk_threshold
        ]
        return safe or list(actions)

    def _q_shortlist(
        self,
        q_values: np.ndarray,
        visit_counts: np.ndarray,
        actions: list[int],
    ) -> tuple[list[int], float]:
        """Return near-optimal Q actions with extra latitude early in learning."""
        q_max = max(float(q_values[action]) for action in actions)
        q_min = min(float(q_values[action]) for action in actions)
        q_span = q_max - q_min
        state_visits = sum(float(visit_counts[action]) for action in actions)
        uncertainty_allowance = self.early_guidance_bonus / np.sqrt(state_visits + 1.0)
        tolerance = (
            max(self.q_absolute_tolerance, self.q_relative_tolerance * q_span)
            + uncertainty_allowance
        )
        shortlist = [
            action
            for action in actions
            if q_max - float(q_values[action]) <= tolerance
        ]
        return shortlist or [self._diagnostic_argmax(q_values, actions)], float(tolerance)

    def select_action(
        self,
        state: dict,
        record_appraisal_counterfactual: bool = True,
        record_compositionality: bool = False,
    ) -> tuple[int, dict]:
        """Select an action using either legacy addition or composed perspectives."""
        stimulus = build_stimulus(state, self.mot, runtime=self.runtime)
        candidates = build_candidates(state, self.mot, runtime=self.runtime)
        perspective_scores = perspective_candidate_scores(
            self.mot,
            stimulus,
            candidates,
            state,
            runtime=self.runtime,
        )
        if self.appraisal_mode == "neutral" or not record_appraisal_counterfactual:
            neutral_perspective_scores = perspective_scores
        else:
            neutral_perspective_scores = perspective_candidate_scores(
                self.mot,
                stimulus,
                candidates,
                state,
                runtime=self.neutral_runtime,
            )
        mot_scores = perspective_scores.consensus
        neutral_mot_scores = neutral_perspective_scores.consensus
        risk_estimates = np.array(
            [
                float(
                    candidate.metadata.get(
                        "hazard_risk", candidate.metadata.get("risk", 0.0)
                    )
                )
                for candidate in candidates
            ],
            dtype=float,
        )

        key = self._encode(state)
        q_values = self._q_values(key)
        visit_counts = self.visit_counts[key]
        exploration_bonus = self.exploration_bonus_weight / np.sqrt(visit_counts + 1.0)
        risk_penalty = self.risk_weight * self.mot.goal("individuation") * risk_estimates
        common_scores = q_values - risk_penalty + exploration_bonus
        regularized_scores = common_scores + self.motivation_weight * mot_scores
        neutral_scores = common_scores + self.motivation_weight * neutral_mot_scores

        exploit_actions = self._valid_actions(
            state,
            avoid_lava=self.mask_lava_on_exploit,
        )
        q_greedy_action = self._diagnostic_argmax(q_values, exploit_actions)
        eligible_actions = self._hard_safety_actions(exploit_actions, risk_estimates)
        safe_q_greedy_action = self._diagnostic_argmax(q_values, eligible_actions)

        task_scores, safety_scores, branch_disagreement, composed_scores = (
            self._composed_preferences(perspective_scores, exploit_actions)
        )
        (
            _,
            _,
            _,
            neutral_composed_scores,
        ) = self._composed_preferences(neutral_perspective_scores, exploit_actions)

        if self.selector_mode == "legacy_additive":
            shortlist = list(eligible_actions)
            effective_q_tolerance = 0.0
            greedy_action = self._diagnostic_argmax(regularized_scores, shortlist)
            neutral_greedy_action = self._diagnostic_argmax(neutral_scores, shortlist)
            selector_scores = regularized_scores
        else:
            shortlist, effective_q_tolerance = self._q_shortlist(
                q_values,
                visit_counts,
                eligible_actions,
            )
            greedy_action = self._diagnostic_argmax(composed_scores, shortlist)
            neutral_greedy_action = self._diagnostic_argmax(
                neutral_composed_scores,
                shortlist,
            )
            selector_scores = composed_scores

        is_exploratory = self.rng.random() < self.epsilon
        if is_exploratory:
            if self.hard_safety:
                action_idx = self.rng.choice(eligible_actions)
            else:
                action_idx = self._select_exploratory_action(state)
        else:
            action_idx = self._argmax_with_random_tie(selector_scores, shortlist)

        compositionality_evaluated = False
        compositionality_error = 0.0
        compositionality_tolerance = 0.0
        compositionality_holds = True
        compositionality_max_goal_error = 0.0
        compositionality_max_goal_coordinate = ""
        compositionality_max_modulator_error = 0.0
        compositionality_max_modulator_coordinate = ""
        compositionality_parallel_action = ""
        compositionality_merged_first_action = ""
        compositionality_action_holds = True
        if (
            record_compositionality
            and self.selector_mode != "legacy_additive"
            and self._decision_count % self.compositionality_audit_interval == 0
        ):
            law_result = measure_gridworld_compositionality(
                state,
                self.mot,
                stimulus,
                candidates,
                runtime=self.runtime,
            )
            compositionality_evaluated = True
            compositionality_error = float(law_result.error)
            compositionality_tolerance = float(law_result.tolerance)
            compositionality_holds = bool(law_result.holds)
            compositionality_max_goal_error = law_result.max_goal_error
            compositionality_max_goal_coordinate = law_result.max_goal_coordinate
            compositionality_max_modulator_error = law_result.max_modulator_error
            compositionality_max_modulator_coordinate = (
                law_result.max_modulator_coordinate
            )
            compositionality_parallel_action = law_result.parallel_action_id
            compositionality_merged_first_action = law_result.merged_first_action_id
            compositionality_action_holds = law_result.action_holds
        self._decision_count += 1

        action, next_state, stimulus, target_state = transition_for_action(
            state,
            self.mot,
            action_idx,
            stimulus=stimulus,
            candidates=candidates,
            runtime=self.runtime,
        )
        self._pending_state = next_state
        self._pending_action = action

        alpha = {
            "appraisal_mode": self.appraisal_mode,
            "appraisal_counterfactual_recorded": record_appraisal_counterfactual,
            "appraisal_changed_action": greedy_action != neutral_greedy_action,
            "appraisal_score_shift": float(np.max(np.abs(mot_scores - neutral_mot_scores))),
            "greedy_action": greedy_action,
            "neutral_greedy_action": neutral_greedy_action,
            "selector_mode": self.selector_mode,
            "q_greedy_action": q_greedy_action,
            "safe_q_greedy_action": safe_q_greedy_action,
            "executed_action": action_idx,
            "q_proposed_hazard_risk": float(risk_estimates[q_greedy_action]),
            "executed_hazard_risk": float(risk_estimates[action_idx]),
            "q_proposed_immediate_lava": bool(
                risk_estimates[q_greedy_action] >= self.hard_safety_risk_threshold
            ),
            "executed_immediate_lava": bool(
                risk_estimates[action_idx] >= self.hard_safety_risk_threshold
            ),
            "q_proposal_blocked": q_greedy_action not in eligible_actions,
            "q_proposed_visit_count": float(visit_counts[q_greedy_action]),
            "q_proposed_value": float(q_values[q_greedy_action]),
            "executed_changed_q_action": action_idx != q_greedy_action,
            "metamo_changed_q_action": greedy_action != q_greedy_action,
            "metamo_changed_safe_q_action": greedy_action != safe_q_greedy_action,
            "hard_safety_intervened": safe_q_greedy_action != q_greedy_action,
            "q_shortlist_size": len(shortlist),
            "effective_q_tolerance": effective_q_tolerance,
            "exploratory": is_exploratory,
            "risk": float(stimulus.hazard_pressure),
            "urgency": float(stimulus.energy_deficit),
            "eu": float(stimulus.goal_proximity),
            "safe_progress": float(stimulus.safe_progress),
            "individuation": float(self.mot.goal("individuation")),
            "transcendence": float(self.mot.goal("transcendence")),
            "target_individuation": float(target_state.goal("individuation")),
            "target_transcendence": float(target_state.goal("transcendence")),
            "q_value": float(q_values[action_idx]),
            "motivation_score": float(mot_scores[action_idx]),
            "task_score": float(task_scores[action_idx]),
            "safety_score": float(safety_scores[action_idx]),
            "task_safety_disagreement": float(branch_disagreement[action_idx]),
            "composed_motivation_score": float(composed_scores[action_idx]),
            "task_score_gain": float(task_scores[greedy_action] - task_scores[q_greedy_action]),
            "safety_score_gain": float(safety_scores[greedy_action] - safety_scores[q_greedy_action]),
            "q_regret": float(q_values[q_greedy_action] - q_values[greedy_action]),
            "exploration_bonus": float(exploration_bonus[action_idx]),
            "risk_penalty": float(risk_penalty[action_idx]),
            "combined_score": float(selector_scores[action_idx]),
            "compositionality_evaluated": compositionality_evaluated,
            "compositionality_error": compositionality_error,
            "compositionality_tolerance": compositionality_tolerance,
            "compositionality_holds": compositionality_holds,
            "compositionality_max_goal_error": compositionality_max_goal_error,
            "compositionality_max_goal_coordinate": (
                compositionality_max_goal_coordinate
            ),
            "compositionality_max_modulator_error": (
                compositionality_max_modulator_error
            ),
            "compositionality_max_modulator_coordinate": (
                compositionality_max_modulator_coordinate
            ),
            "compositionality_parallel_action": compositionality_parallel_action,
            "compositionality_merged_first_action": (
                compositionality_merged_first_action
            ),
            "compositionality_action_holds": compositionality_action_holds,
        }
        return action_idx, alpha

    def commit_motivational_transition(self) -> None:
        """Commit the already-selected MetaMo transition without learning Q."""
        if self._pending_state is not None:
            self.mot = self._pending_state
            self._pending_state = None
            self._pending_action = None

    def update(
        self,
        state: dict,
        action: int,
        reward: float,
        next_state: dict,
        done: bool,
        event: Optional[str],
        alpha: dict,
    ):
        """Apply the same shaped tabular Q update used by the baseline."""
        s = self._encode(state)
        ns = self._encode(next_state)
        q_values = self._q_values(s)
        next_q_values = self._q_values(ns)
        shaped_reward = self._shape_reward(state, reward, next_state)
        if done:
            next_value = 0.0
        else:
            next_actions = self._valid_actions(next_state)
            next_value = max(next_q_values[next_action] for next_action in next_actions)
        td_target = shaped_reward + self.gamma * next_value
        td_error = td_target - q_values[action]
        q_values[action] += self.alpha * td_error
        self.visit_counts[s][action] += 1.0

        self.commit_motivational_transition()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
