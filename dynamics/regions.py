from dataclasses import dataclass
from typing import List

import numpy as np

from core.state import MotivationalState
from dynamics.stability import is_in_safe_region, project_to_safe_region


@dataclass(frozen=True)
class IdealRegion:
    """
    Practical representation of MetaMo's reachable ideal region I.
    """

    center: MotivationalState
    radius: float = 0.05
    require_safe: bool = True

    def __post_init__(self):
        if self.radius <= 0.0:
            raise ValueError("IdealRegion radius must be positive")

    def distance_to(self, state: MotivationalState) -> float:
        return self.center.distance_to(state)

    def contains(self, state: MotivationalState) -> bool:
        if self.require_safe and not is_in_safe_region(state):
            return False
        return self.distance_to(state) <= self.radius

    def project(self, state: MotivationalState) -> MotivationalState:
        """
        Project a state into the ideal ball, then into the MetaMo safe region.
        """
        if self.contains(state):
            return state.copy()

        delta_G = state.G - self.center.G
        delta_M = state.M - self.center.M
        distance = self.distance_to(state)

        if distance == 0.0:
            projected = self.center.copy()
        else:
            scale = min(1.0, self.radius / distance)
            projected = MotivationalState(
                G=np.clip(self.center.G + delta_G * scale, 0.0, 1.0),
                M=np.clip(self.center.M + delta_M * scale, 0.0, 1.0),
            )

        if self.require_safe:
            return project_to_safe_region(projected)
        return projected


@dataclass(frozen=True)
class ReachableRegion:
    """
    Finite-dimensional approximation to a tubular reachable region.
    """

    ideal: IdealRegion
    max_step_distance: float = 0.1

    def __post_init__(self):
        if self.max_step_distance <= 0.0:
            raise ValueError("max_step_distance must be positive")

    def waypoints(self, start: MotivationalState, target: MotivationalState) -> List[MotivationalState]:
        """
        Build a linear thick-path approximation from start to the projected target.
        """
        projected_target = self.ideal.project(target)
        total_distance = start.distance_to(projected_target)
        if total_distance == 0.0:
            return [start.copy()]

        steps = max(1, int(np.ceil(total_distance / self.max_step_distance)))
        path = []
        for idx in range(steps + 1):
            alpha = idx / steps
            path.append(MotivationalState(
                G=((1.0 - alpha) * start.G) + (alpha * projected_target.G),
                M=((1.0 - alpha) * start.M) + (alpha * projected_target.M),
            ))
        return path

    def is_reachable(self, start: MotivationalState, target: MotivationalState) -> bool:
        """
        Checks whether a sampled path stays safe and ends inside the ideal region.
        """
        path = self.waypoints(start, target)
        if self.ideal.require_safe and any(not is_in_safe_region(point) for point in path):
            return False
        return self.ideal.contains(path[-1])
