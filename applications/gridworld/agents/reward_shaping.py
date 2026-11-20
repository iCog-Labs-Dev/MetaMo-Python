"""Shared reward shaping for GridWorld Q-learning agents."""

from __future__ import annotations

from dataclasses import dataclass

from applications.gridworld.agents.q_features import manhattan_distance


@dataclass(frozen=True)
class RewardShapingConfig:
    """Weights for optional task reward shaping."""

    progress_weight: float = 8.0
    lava_penalty: float = 80.0
    boundary_penalty: float = 5.0
    danger_penalty: float = 2.0
    danger_distance: int = 2


def shape_reward(
    state: dict,
    reward: float,
    next_state: dict,
    config: RewardShapingConfig = RewardShapingConfig(),
) -> float:
    """
    Apply shared shaping used by baseline and optional MetaMo conditions.

    The environment reward remains the source of truth. Shaping is an
    experimental condition that should be applied symmetrically when used.
    """
    shaped_reward = float(reward)
    old_mineral_pos = state["mineral_pos"]
    collected_mineral = next_state["pos"] == old_mineral_pos

    if not collected_mineral:
        before_dist = manhattan_distance(state["pos"], old_mineral_pos)
        after_dist = manhattan_distance(next_state["pos"], old_mineral_pos)
        shaped_reward += config.progress_weight * (before_dist - after_dist)

    if next_state.get("in_lava", False):
        shaped_reward -= config.lava_penalty

    if next_state["pos"] == state["pos"] and not collected_mineral:
        shaped_reward -= config.boundary_penalty

    lava_distance = int(next_state.get("lava_distance", config.danger_distance + 2))
    danger_depth = max(0, config.danger_distance + 1 - lava_distance)
    shaped_reward -= config.danger_penalty * danger_depth

    return shaped_reward


def learning_reward(
    state: dict,
    reward: float,
    next_state: dict,
    mode: str = "shaped",
    config: RewardShapingConfig = RewardShapingConfig(),
) -> float:
    """Return the reward used by a learner under a named reward mode."""
    if mode == "raw":
        return float(reward)
    if mode == "shaped":
        return shape_reward(state, reward, next_state, config)
    raise ValueError(f"unknown reward mode: {mode}")
