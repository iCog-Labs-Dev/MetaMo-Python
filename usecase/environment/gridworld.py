"""
10x10 GridWorld with randomized lava hazards and mixed safe/danger mineral spawns.
"""

import random
from typing import Literal

GRID_SIZE = 10
MAX_STEPS = 50
LAVA_CELLS = {(8, 8), (8, 9), (9, 8), (9, 9)}
LAVA_CELL_COUNT = 4
MINERAL_SPAWN_BAND = 2
DANGER_MINERAL_PROBABILITY = 0.55
AGENT_START = (0, 0)

REWARD_MINERAL = 100.0
REWARD_LAVA = -15.0
REWARD_STEP = -0.5
REWARD_BOUNDARY = -1.0


class GridWorld:
    """Seeded 10x10 grid environment with autonomous mineral collection."""

    def __init__(
        self,
        seed: int = 42,
        max_steps: int = MAX_STEPS,
        lava_cell_count: int = LAVA_CELL_COUNT,
        mineral_spawn_band: int = MINERAL_SPAWN_BAND,
        danger_mineral_probability: float = DANGER_MINERAL_PROBABILITY,
    ):
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.lava_cell_count = lava_cell_count
        self.mineral_spawn_band = mineral_spawn_band
        self.danger_mineral_probability = danger_mineral_probability
        self.reset()

    def reset(self):
        self.agent_pos = list(AGENT_START)
        self.lava_cells = self._generate_lava_cells()
        self.mineral_pos = list(AGENT_START)
        self.energy = 100.0
        self.step_count = 0
        self.minerals_collected = 0
        self.minerals_spawned = 0
        self.done = False
        self._respawn_mineral()
        return self._get_state()

    def step(self, action: int):
        if self.done:
            return self._get_state(), 0.0, True, {}

        dx, dy = self._action_to_delta(action)
        nx = self.agent_pos[0] + dx
        ny = self.agent_pos[1] + dy

        reward = REWARD_STEP
        info = {"event": None}

        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            self.agent_pos = [nx, ny]
        else:
            reward += REWARD_BOUNDARY
            info["event"] = "boundary"

        if tuple(self.agent_pos) in self.lava_cells:
            self.energy += REWARD_LAVA
            reward += REWARD_LAVA
            info["event"] = "lava"

        if self.agent_pos == self.mineral_pos:
            self.energy = min(100.0, self.energy + REWARD_MINERAL)
            reward += REWARD_MINERAL
            self.minerals_collected += 1
            info["event"] = "mineral"
            self._respawn_mineral()

        self.step_count += 1
        if self.energy <= 0:
            self.done = True
            info["event"] = "dead"
        if self.step_count >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done, info

    def _get_state(self) -> dict:
        row, col = self.agent_pos
        mrow, mcol = self.mineral_pos
        in_lava = tuple(self.agent_pos) in self.lava_cells
        lava_distance = self._distance_to_nearest_lava(self.agent_pos)
        return {
            "pos": tuple(self.agent_pos),
            "mineral_pos": tuple(self.mineral_pos),
            "energy": self.energy,
            "in_lava": in_lava,
            "lava_distance": lava_distance,
            "lava_cells": tuple(sorted(self.lava_cells)),
            "dx_mineral": mcol - col,
            "dy_mineral": mrow - row,
            "step": self.step_count,
        }

    def _respawn_mineral(self):
        prefer_danger = self.rng.random() < self.danger_mineral_probability
        preferred_zone = "danger" if prefer_danger else "safe"
        backup_zone = "safe" if prefer_danger else "danger"
        spawn_pool = (
            self._mineral_spawn_candidates(preferred_zone)
            or self._mineral_spawn_candidates(backup_zone)
            or self._mineral_spawn_candidates("any")
        )
        if spawn_pool:
            self.mineral_pos = list(self.rng.choice(spawn_pool))
            self.minerals_spawned += 1

    def _mineral_spawn_candidates(
        self,
        zone: Literal["danger", "safe", "any"],
    ) -> list[tuple[int, int]]:
        cells = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = (r, c)
                if cell in self.lava_cells:
                    continue
                if [r, c] == self.agent_pos:
                    continue
                dist = self._distance_to_nearest_lava(cell)
                if zone == "danger" and dist > self.mineral_spawn_band:
                    continue
                if zone == "safe" and dist <= self.mineral_spawn_band:
                    continue
                cells.append(cell)
        return cells

    def _generate_lava_cells(self) -> set[tuple[int, int]]:
        candidates = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = (r, c)
                if cell == AGENT_START:
                    continue
                if abs(r - AGENT_START[0]) + abs(c - AGENT_START[1]) <= 2:
                    continue
                candidates.append(cell)

        if len(candidates) < self.lava_cell_count:
            return set(LAVA_CELLS)
        return set(self.rng.sample(candidates, self.lava_cell_count))

    def _distance_to_nearest_lava(self, cell: tuple[int, int]) -> int:
        return min(abs(cell[0] - lava[0]) + abs(cell[1] - lava[1]) for lava in self.lava_cells)

    @staticmethod
    def _action_to_delta(action: int):
        return [(-1, 0), (1, 0), (0, -1), (0, 1)][action]

    def is_lava_cell(self, cell: tuple[int, int]) -> bool:
        return cell in self.lava_cells
