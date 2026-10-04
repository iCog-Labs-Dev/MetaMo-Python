from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from applications.gridworld.config import (
    ACTION_DELTAS,
    ACTION_IDS,
    APPRAISAL_DELTA_SCALE,
    FALLBACK_LAVA_CELLS,
    GRID_SIZE,
    MAX_MANHATTAN_DISTANCE,
    MAX_STEPS,
    hazard_pressure,
)
from applications.gridworld.schema import GRIDWORLD_SCHEMA
from applications.gridworld.stimulus import GridWorldStimulus
from core.features import GoalChangeFeedback, StimulusFeatures
from core.schema import MotivationSchema
from core.state import Action, MotivationalState
from magus.profile import DecisionProfile
from openpsi.profile import AppraisalProfile


class GridWorldAppraisalProfile(AppraisalProfile):
    """Small, bounded rule-based appraisal tailored to this task."""

    def stimulus_features(self, state, stimulus):
        if isinstance(stimulus, StimulusFeatures):
            return stimulus
        if not isinstance(stimulus, GridWorldStimulus):
            raise TypeError("gridworld appraisal expects GridWorldStimulus")
        return StimulusFeatures(
            goal_proximity=stimulus.goal_proximity,
            hazard_pressure=stimulus.hazard_pressure,
            energy_deficit=stimulus.energy_deficit,
            time_pressure=stimulus.time_pressure,
            safe_progress=stimulus.safe_progress,
            safe_mobility=stimulus.safe_mobility,
        )

    def core_modulator_deltas(self, state, features):
        proximity = features.numeric("goal_proximity")
        hazard = features.numeric("hazard_pressure")
        deficit = features.numeric("energy_deficit")
        time = features.numeric("time_pressure")
        progress = features.numeric("safe_progress")
        mobility = features.numeric("safe_mobility")
        scale = APPRAISAL_DELTA_SCALE
        return {
            "valence": scale * (0.45 * proximity + 0.45 * progress - 0.60 * hazard - 0.20 * deficit),
            "arousal": scale * (0.50 * hazard + 0.25 * deficit + 0.25 * time),
            "approach": scale * (0.65 * progress + 0.25 * proximity - 0.75 * hazard),
            "resolution": scale * (0.65 * progress + 0.25 * mobility - 0.35 * hazard),
            "threshold": scale * (0.70 * hazard + 0.30 * deficit + 0.20 * (1.0 - mobility)),
            "securing": scale * (0.75 * hazard + 0.35 * deficit + 0.25 * (1.0 - mobility)),
        }

    def goal_change_feedback(self, state, stimulus, features):
        hazard = features.numeric("hazard_pressure")
        deficit = features.numeric("energy_deficit")
        time = features.numeric("time_pressure")
        progress = features.numeric("safe_progress")
        mobility = features.numeric("safe_mobility")
        return GoalChangeFeedback(
            individuation_target=np.clip(0.50 + 0.30 * hazard + 0.20 * deficit + 0.10 * (1.0 - mobility), 0.0, 1.0),
            transcendence_target=np.clip(0.45 + 0.35 * progress + 0.15 * time - 0.35 * hazard, 0.0, 1.0),
            lava_exposure_target=hazard,
            boundary_collision_target=1.0 - mobility,
        )


    def goal_change_feedback(self, state, stimulus, features):
        return GoalChangeFeedback()


@dataclass(frozen=True)
class GridWorldProfile:
    """Translate GridWorld state and actions into MetaMo application data."""

    schema: MotivationSchema = GRIDWORLD_SCHEMA
    lava_cells: tuple[tuple[int, int], ...] = FALLBACK_LAVA_CELLS
    action_ids: tuple[str, ...] = ACTION_IDS
    deltas: tuple[tuple[int, int], ...] = ACTION_DELTAS
    grid_size: int = GRID_SIZE

    def create_initial_state(self) -> MotivationalState:
        state = MotivationalState(
            G=np.zeros(self.schema.num_goals, dtype=float),
            M=np.full(self.schema.num_modulators, 0.5, dtype=float),
            schema=self.schema,
        )
        for name, value in {
            "individuation": 0.70,
            "transcendence": 0.55,
            "mineral_acquisition": 0.75,
            "energy_preservation": 0.65,
            "navigation_efficiency": 0.45,
            "lava_exposure": 0.85,
            "boundary_collision": 0.35,
        }.items():
            state.set_goal(name, value)
        return state

    def safety_threshold(self, mot_state):
        return mot_state.modulator("threshold")

    def arousal(self, mot_state):
        return mot_state.modulator("arousal")

    @staticmethod
    def l1_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def lava_cells_from_state(self, env_state):
        return tuple(env_state.get("lava_cells", self.lava_cells))

    def distance_to_lava(self, pos, lava_cells):
        return min(self.l1_distance(pos, lava) for lava in lava_cells)

    @staticmethod
    def is_lava_cell(pos, lava_cells):
        return pos in lava_cells

    def project_move(self, env_state, action):
        row, col = env_state["pos"]
        dr, dc = self.deltas[action]
        proposed = (row + dr, col + dc)
        if 0 <= proposed[0] < self.grid_size and 0 <= proposed[1] < self.grid_size:
            return proposed
        return (row, col)

    def _safe_move_fraction(self, pos, lava_cells):
        safe = 0
        for dr, dc in self.deltas:
            candidate = (pos[0] + dr, pos[1] + dc)
            if (
                0 <= candidate[0] < self.grid_size
                and 0 <= candidate[1] < self.grid_size
                and candidate not in lava_cells
            ):
                safe += 1
        return safe / len(self.deltas)

    def build_stimulus(self, env_state, mot_state: Optional[MotivationalState] = None):
        distance = abs(env_state["dx_mineral"]) + abs(env_state["dy_mineral"])
        lava_cells = self.lava_cells_from_state(env_state)
        lava_distance = env_state.get("lava_distance", self.distance_to_lava(env_state["pos"], lava_cells))
        proximity = 1.0 - distance / MAX_MANHATTAN_DISTANCE
        safe_progress = proximity * (1.0 - hazard_pressure(lava_distance))
        return GridWorldStimulus(
            goal_proximity=float(np.clip(proximity, 0.0, 1.0)),
            hazard_pressure=hazard_pressure(lava_distance),
            energy_deficit=float(np.clip(1.0 - env_state.get("energy", 100.0) / 100.0, 0.0, 1.0)),
            time_pressure=float(np.clip(env_state.get("step", 0) / MAX_STEPS, 0.0, 1.0)),
            safe_progress=float(np.clip(safe_progress, 0.0, 1.0)),
            safe_mobility=self._safe_move_fraction(env_state["pos"], lava_cells),
        )

    def make_local_candidate(self, env_state, action):
        lava_cells = self.lava_cells_from_state(env_state)
        pos = env_state["pos"]
        next_pos = self.project_move(env_state, action)
        boundary_hit = float(next_pos == pos and (
            pos[0] + self.deltas[action][0] < 0 or pos[0] + self.deltas[action][0] >= self.grid_size
            or pos[1] + self.deltas[action][1] < 0 or pos[1] + self.deltas[action][1] >= self.grid_size
        ))
        dist_now = self.l1_distance(pos, env_state["mineral_pos"])
        next_dist = self.l1_distance(next_pos, env_state["mineral_pos"])
        progress = float(np.clip((dist_now - next_dist + 1.0) / 2.0, 0.0, 1.0))
        collected = float(next_pos == env_state["mineral_pos"])
        next_lava_dist = self.distance_to_lava(next_pos, lava_cells)
        current_lava_dist = env_state.get("lava_distance", self.distance_to_lava(pos, lava_cells))
        risk = hazard_pressure(next_lava_dist)
        safe_exit_delta = float(np.clip(next_lava_dist - current_lava_dist, -1.0, 1.0))
        safe_exit_gain = float(np.clip((next_lava_dist - current_lava_dist + 1.0) / 2.0, 0.0, 1.0))
        deficit = float(np.clip(1.0 - env_state.get("energy", 100.0) / 100.0, 0.0, 1.0))
        energy_impact = float(
            np.clip(
                collected * deficit - 0.15 * float(self.is_lava_cell(next_pos, lava_cells)),
                -1.0,
                1.0,
            )
        )

        correlations = np.zeros(self.schema.num_goals, dtype=float)
        values = {
            "individuation": 1.0 - risk,
            "transcendence": progress * (1.0 - risk),
            "mineral_acquisition": np.clip(0.70 * progress + 0.30 * collected, 0.0, 1.0),
            "energy_preservation": np.clip(
                energy_impact + 0.25 * safe_exit_delta,
                -1.0,
                1.0,
            ),
            "navigation_efficiency": np.clip(0.60 * progress + 0.30 * safe_exit_gain - 0.50 * boundary_hit, 0.0, 1.0),
            "lava_exposure": risk,
            "boundary_collision": boundary_hit,
        }
        for name, value in values.items():
            correlations[self.schema.goal_index(name)] = value

        return Action(
            id=self.action_ids[action],
            goal_correlations=correlations,
            delta_g=np.zeros(self.schema.num_goals, dtype=float),
            schema=self.schema,
            metadata={
                "risk": risk,
                "hazard_risk": risk,
                "opportunity": progress * (1.0 - risk),
                "mineral_progress": progress,
                "mineral_collected": collected,
                "boundary_hit": boundary_hit,
                "safe_exit_gain": safe_exit_gain,
                "safe_exit_delta": safe_exit_delta,
                "energy_impact": energy_impact,
                "next_lava_distance": next_lava_dist,
            },
        )

    def build_candidates(self, env_state, mot_state=None) -> List[Action]:
        return [self.make_local_candidate(env_state, action) for action in range(len(self.deltas))]

    def build_consensus_states(self, env_state, mot_state, stimulus):
        hazard = stimulus.hazard_pressure
        deficit = stimulus.energy_deficit
        opportunity = stimulus.safe_progress

        safety = mot_state.copy()
        safety.set_goal("individuation", np.clip(safety.goal("individuation") + 0.10 * hazard + 0.05 * deficit, 0.0, 1.0))
        safety.set_goal("energy_preservation", np.clip(safety.goal("energy_preservation") + 0.10 * hazard + 0.08 * deficit, 0.0, 1.0))
        safety.set_goal("lava_exposure", np.clip(safety.goal("lava_exposure") + 0.12 * hazard, 0.0, 1.0))
        safety.set_goal("transcendence", np.clip(safety.goal("transcendence") - 0.06 * hazard, 0.0, 1.0))

        growth = mot_state.copy()
        growth.set_goal("transcendence", np.clip(growth.goal("transcendence") + 0.14 * opportunity, 0.0, 1.0))
        growth.set_goal("mineral_acquisition", np.clip(growth.goal("mineral_acquisition") + 0.12 * opportunity, 0.0, 1.0))
        growth.set_goal("navigation_efficiency", np.clip(growth.goal("navigation_efficiency") + 0.10 * opportunity, 0.0, 1.0))
        growth.set_goal("individuation", np.clip(growth.goal("individuation") + 0.03 * hazard, 0.0, 1.0))
        return safety, growth


GRIDWORLD_PROFILE = GridWorldProfile()
GRIDWORLD_APPRAISAL_PROFILE = GridWorldAppraisalProfile(name="gridworld_openpsi")
GRIDWORLD_DECISION_PROFILE = DecisionProfile(
    name="gridworld_magus",
    schema=GRIDWORLD_SCHEMA,
    goal_modulators={
        "mineral_acquisition": ("approach", "valence", "resolution"),
        "energy_preservation": ("threshold", "securing"),
        "navigation_efficiency": ("resolution", "approach"),
    },
    individuation_goal_names=("energy_preservation",),
    transcendence_goal_names=("mineral_acquisition",),
    balanced_goal_names=("navigation_efficiency",),
    anti_goal_penalty_weights={"lava_exposure": 1.8, "boundary_collision": 0.6},
    overgoal_target_features={
        "individuation": "individuation_target",
        "transcendence": "transcendence_target",
    },
    anti_goal_target_features={
        "lava_exposure": "lava_exposure_target",
        "boundary_collision": "boundary_collision_target",
    },
    overgoal_delta_scale=0.01,
    anti_goal_delta_scale=0.02,
)
