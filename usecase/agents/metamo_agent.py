"""
MetaMo-enhanced gridworld agent that reasons with the shared MetaMo pseudo-bimonad.
"""

import numpy as np
from typing import Optional

from core.state import Action, MotivationalState
from core.config import G_IND, G_TRANS
from metamo.state import create_initial_motivational_state
from metamo.core import (
    build_candidates,
    build_stimulus,
    consensus_candidate_scores,
    in_safe_region,
    transition_for_action,
)


class MetaMoAgent:
    """Autonomous gridworld agent with root MetaMo motivational reasoning."""

    ACTIONS = 4

    def __init__(
        self,
        grid_size: int = 10,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.10,
        epsilon_decay: float = 0.995,
        seed: int = 42,
        motivation_weight: float = 6.0,
        risk_weight: float = 4.0,
        exploration_bonus_weight: float = 1.25,
        use_llm_narration: bool = False,
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
        self.rng = np.random.default_rng(seed)

        self.mot = create_initial_motivational_state()
        self._pending_state: Optional[MotivationalState] = None
        self._pending_action: Optional[Action] = None
        self.q_table = np.zeros((grid_size, grid_size, grid_size, grid_size, self.ACTIONS))
        self.visit_counts = np.zeros_like(self.q_table)

        self.log_alpha: list[dict] = []
        self.log_mot: list[MotivationalState] = []
        self.log_srv: list[bool] = []

        self.assistant = None

    def reset_episode(self):
        self.mot = create_initial_motivational_state()
        self._pending_state = None
        self._pending_action = None
        self.log_alpha = []
        self.log_mot = []
        self.log_srv = []

    def _encode(self, state: dict) -> tuple[int, int, int, int]:
        ar, ac = state["pos"]
        mr, mc = state["mineral_pos"]
        return (ar, ac, mr, mc)

    def select_action(self, state: dict) -> tuple[int, dict]:
        stimulus = build_stimulus(state, self.mot)
        candidates = build_candidates(state, self.mot)
        mot_scores = consensus_candidate_scores(self.mot, stimulus, candidates, state)
        risk_estimates = np.array([candidate.risk_estimate for candidate in candidates], dtype=float)
        q_values = self.q_table[self._encode(state)]
        visit_counts = self.visit_counts[self._encode(state)]
        exploration_bonus = self.exploration_bonus_weight / np.sqrt(visit_counts + 1.0)
        regularized_scores = (
            q_values
            + self.motivation_weight * mot_scores
            - self.risk_weight * self.mot.G[G_IND] * risk_estimates
            + exploration_bonus
        )

        if self.rng.random() < self.epsilon:
            action_idx = int(self.rng.integers(0, self.ACTIONS))
        else:
            action_idx = int(np.argmax(regularized_scores))

        action, next_state, stimulus, target_state = transition_for_action(
            state,
            self.mot,
            action_idx,
            stimulus=stimulus,
            candidates=candidates,
        )
        self._pending_state = next_state
        self._pending_action = action

        alpha = {
            "risk": float(stimulus.risk),
            "urgency": float(np.clip(1.0 - state["energy"] / 100.0, 0.0, 1.0)),
            "eu": float(np.clip(1.0 - (abs(state["dx_mineral"]) + abs(state["dy_mineral"])) / 18.0, 0.0, 1.0)),
            "individuation": float(self.mot.G[G_IND]),
            "transcendence": float(self.mot.G[G_TRANS]),
            "target_individuation": float(target_state.G[G_IND]),
            "target_transcendence": float(target_state.G[G_TRANS]),
            "q_value": float(q_values[action_idx]),
            "motivation_score": float(mot_scores[action_idx]),
            "exploration_bonus": float(exploration_bonus[action_idx]),
            "combined_score": float(regularized_scores[action_idx]),
        }

        self.log_alpha.append(alpha)
        self.log_mot.append(self.mot.copy())
        self.log_srv.append(not in_safe_region(self.mot))

        return action_idx, alpha

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
        s = self._encode(state)
        ns = self._encode(next_state)
        td_target = reward + (0.0 if done else self.gamma * np.max(self.q_table[ns]))
        td_error = td_target - self.q_table[s + (action,)]
        self.q_table[s + (action,)] += self.alpha * td_error
        self.visit_counts[s + (action,)] += 1.0

        if self._pending_state is not None:
            self.mot = self._pending_state
            self._pending_state = None
            self._pending_action = None

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def explain_action(self, user_text: str) -> str:
        if self.assistant is not None and self._pending_action is not None:
            try:
                return self.assistant.generate_final_response(user_text, self._pending_action, self.mot)
            except Exception:
                return f"Chosen action: {self._pending_action.id}."
        return "Chosen action executed by the autonomous MetaMo controller."

    @property
    def srv_rate(self) -> float:
        if not self.log_srv:
            return 0.0
        return sum(self.log_srv) / len(self.log_srv)

    @property
    def mean_arousal(self) -> float:
        if not self.log_mot:
            return 0.0
        return float(np.mean([m.M[1] for m in self.log_mot])) if self.log_mot else 0.0

    @property
    def mean_safety(self) -> float:
        if not self.log_mot:
            return 0.0
        return float(np.mean([m.M[4] for m in self.log_mot])) if self.log_mot else 0.0
