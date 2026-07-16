from dataclasses import dataclass
from typing import Dict

import numpy as np

from core.config import (NUM_GOALS, NUM_MODULATORS)


@dataclass
class MotivationalState:
    """
    Represents the motivational state object X = G x M in the MetaMo category.
    """

    G: np.ndarray  # Vector of goal intensities/weights.
    M: np.ndarray  # Vector of continuous OpenPsi modulators.

    def __post_init__(self):
        """Ensure vectors are initialized with the correct dimensions."""
        if self.G.shape[0] != NUM_GOALS:
            raise ValueError(f"Goal vector must have length {NUM_GOALS}")
        if self.M.shape[0] != NUM_MODULATORS:
            raise ValueError(f"Modulator vector must have length {NUM_MODULATORS}")

    def copy(self) -> "MotivationalState":
        """Creates a deep copy of the state for safe functional updates."""
        return MotivationalState(self.G.copy(), self.M.copy())

    def distance_to(self, other: "MotivationalState") -> float:
        """
        Calculates the distance d(x, y) between two states.
        Essential for checking the contractive update law: d(F(x), F(y)) <= c*d(x,y) + epsilon.
        """
        dist_G = np.linalg.norm(self.G - other.G)
        dist_M = np.linalg.norm(self.M - other.M)
        return dist_G + dist_M


@dataclass
class Stimulus:
    """
    Represents an external or internal event (s) passed to the Appraisal Comonad (Psi).
    """

    novelty: float  # Triggers arousal and approach.
    conduciveness: (
        float  # Goal conduciveness; triggers valence and resolution.
    )
    risk: float  # Triggers threshold and securing (caution).
    effort: float = 0.0  # Cognitive or physical effort required.


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

    def __post_init__(self):
        if self.goal_correlations.shape[0] != NUM_GOALS:
            raise ValueError(f"Correlations vector must have length {NUM_GOALS}")
        if self.delta_g.shape[0] != NUM_GOALS:
            raise ValueError(f"Delta G vector must have length {NUM_GOALS}")
