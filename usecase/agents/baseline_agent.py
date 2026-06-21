"""
Baseline RL agent: tabular Q-learning with no motivational layer.
Greedy task-seeker 
"""

import numpy as np
import random


class BaselineAgent:
    """
    Tabular Q-learning agent.

    State space is encoded as (agent_row, agent_col, mineral_row, mineral_col).
    No internal motivational state — purely reward-driven.
    """

    ACTIONS = 4  

    def __init__(
        self,
        grid_size:    int   = 10,
        alpha:        float = 0.1,   
        gamma:        float = 0.95, 
        epsilon:      float = 1.0,   
        epsilon_min:  float = 0.05,
        epsilon_decay:float = 0.995,
        seed:         int   = 42,
    ):
        self.grid_size     = grid_size
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng           = random.Random(seed)

        # Q-table: [agent_row, agent_col, min_row, min_col, action]
        self.q_table = np.zeros((grid_size, grid_size,
                                 grid_size, grid_size,
                                 self.ACTIONS))

    
    def _encode(self, state: dict) -> tuple:
        ar, ac = state["pos"]
        mr, mc = state["mineral_pos"]
        return (ar, ac, mr, mc)

    def select_action(self, state: dict) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.ACTIONS - 1)
        s = self._encode(state)
        return int(np.argmax(self.q_table[s]))

    def update(self, state: dict, action: int, reward: float,
               next_state: dict, done: bool):
        s  = self._encode(state)
        ns = self._encode(next_state)
        td_target = reward + (0.0 if done else self.gamma * np.max(self.q_table[ns]))
        td_error  = td_target - self.q_table[s + (action,)]
        self.q_table[s + (action,)] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    # does noting currently but we might use it for future if we want to use and reset some internal variables 
    def reset_episode(self):
        pass
