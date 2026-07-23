from dataclasses import dataclass

import numpy as np

from core.schema import DEFAULT_MOTIVATION_SCHEMA, MotivationSchema


@dataclass
class MotivationalState:
    """
    Represents the motivational state object X = G x M in the MetaMo category.
    """

    G: np.ndarray  # Vector of goal intensities/weights.
    M: np.ndarray  # Vector of continuous OpenPsi modulators.
    schema: MotivationSchema = DEFAULT_MOTIVATION_SCHEMA

    def __post_init__(self):
        """Ensure vectors are initialized with the correct dimensions."""
        self.G = np.asarray(self.G, dtype=float)
        self.M = np.asarray(self.M, dtype=float)

        if self.G.ndim != 1:
            raise ValueError("Goal vector must be one-dimensional")
        if self.M.ndim != 1:
            raise ValueError("Modulator vector must be one-dimensional")
        if self.G.shape[0] != self.schema.num_goals:
            raise ValueError(f"Goal vector must have length {self.schema.num_goals}")
        if self.M.shape[0] != self.schema.num_modulators:
            raise ValueError(f"Modulator vector must have length {self.schema.num_modulators}")

    def copy(self) -> "MotivationalState":
        """Creates a deep copy of the state for safe functional updates."""
        return MotivationalState(self.G.copy(), self.M.copy(), schema=self.schema)

    def distance_to(self, other: "MotivationalState") -> float:
        """
        Calculates the distance d(x, y) between two states.
        Essential for checking the contractive update law: d(F(x), F(y)) <= c*d(x,y) + epsilon.
        """
        if self.schema != other.schema:
            raise ValueError("Cannot measure distance between states with different motivation schemas")
        if self.G.shape != other.G.shape or self.M.shape != other.M.shape:
            raise ValueError("Cannot measure distance between states with different dimensions")
        dist_G = np.linalg.norm(self.G - other.G)
        dist_M = np.linalg.norm(self.M - other.M)
        return dist_G + dist_M

    def goal_index(self, name: str) -> int:
        return self.schema.goal_index(name)

    def modulator_index(self, name: str) -> int:
        return self.schema.modulator_index(name)

    def goal(self, name: str) -> float:
        return float(self.G[self.goal_index(name)])

    def set_goal(self, name: str, value: float) -> None:
        self.G[self.goal_index(name)] = value

    def anti_goal(self, name: str) -> float:
        if not self.schema.is_anti_goal(name):
            raise KeyError(f"unknown anti-goal name: {name}")
        return self.goal(name)

    def set_anti_goal(self, name: str, value: float) -> None:
        if not self.schema.is_anti_goal(name):
            raise KeyError(f"unknown anti-goal name: {name}")
        self.set_goal(name, value)

    def modulator(self, name: str) -> float:
        return float(self.M[self.modulator_index(name)])

    def set_modulator(self, name: str, value: float) -> None:
        self.M[self.modulator_index(name)] = value

@dataclass
class Action:
    """
    Represents a candidate action or inference rule evaluated by the Decision Monad (D).
    """

    id: str
    # Measures alignment (corr or rel) with each primary goal.
    goal_correlations: np.ndarray
    # Estimates potential ethical breach or operational risk.
    risk_estimate: float
    # Expected modification to the goal vector (Delta G) if selected.
    delta_g: np.ndarray
    schema: MotivationSchema = DEFAULT_MOTIVATION_SCHEMA

    def __post_init__(self):
        self.goal_correlations = np.asarray(self.goal_correlations, dtype=float)
        self.delta_g = np.asarray(self.delta_g, dtype=float)

        if self.goal_correlations.ndim != 1:
            raise ValueError("Correlations vector must be one-dimensional")
        if self.delta_g.ndim != 1:
            raise ValueError("Delta G vector must be one-dimensional")
        if self.goal_correlations.shape[0] != self.schema.num_goals:
            raise ValueError(f"Correlations vector must have length {self.schema.num_goals}")
        if self.delta_g.shape[0] != self.schema.num_goals:
            raise ValueError(f"Delta G vector must have length {self.schema.num_goals}")

    def goal_correlation(self, name: str) -> float:
        return float(self.goal_correlations[self.schema.goal_index(name)])

    def goal_delta(self, name: str) -> float:
        return float(self.delta_g[self.schema.goal_index(name)])
