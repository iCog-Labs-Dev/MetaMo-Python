from dataclasses import dataclass
from typing import Mapping

import numpy as np

from core.schema import MotivationSchema
from core.state import MotivationalState


def _clip_factor(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


@dataclass(frozen=True)
class GoalCompatibilityMatrix:
    """
    Generic MIC helper for goal-goal compatibility.

    weights[i, j] says how much active goal j supports or suppresses goal i.
    The resulting factor is neutral at 1.0, above 1.0 for support, and below
    1.0 for suppression.
    """

    schema: MotivationSchema
    weights: np.ndarray
    influence_scale: float = 0.25
    min_factor: float = 0.0
    max_factor: float = 2.0

    def __post_init__(self):
        weights = np.asarray(self.weights, dtype=float)
        expected_shape = (self.schema.num_goals, self.schema.num_goals)
        if weights.shape != expected_shape:
            raise ValueError(
                f"compatibility weights must have shape {expected_shape}"
            )
        if self.influence_scale < 0.0:
            raise ValueError("influence_scale must be non-negative")
        if self.min_factor > self.max_factor:
            raise ValueError("min_factor must be less than or equal to max_factor")
        object.__setattr__(self, "weights", weights)

    @classmethod
    def neutral(cls, schema: MotivationSchema) -> "GoalCompatibilityMatrix":
        return cls(schema=schema, weights=np.zeros((schema.num_goals, schema.num_goals)))

    @classmethod
    def from_goal_pairs(
        cls,
        schema: MotivationSchema,
        weights: Mapping[tuple[str, str], float],
        *,
        symmetric: bool = False,
        influence_scale: float = 0.25,
        min_factor: float = 0.0,
        max_factor: float = 2.0,
    ) -> "GoalCompatibilityMatrix":
        matrix = np.zeros((schema.num_goals, schema.num_goals), dtype=float)
        for (goal_name, context_goal_name), weight in weights.items():
            goal_idx = schema.goal_index(goal_name)
            context_idx = schema.goal_index(context_goal_name)
            matrix[goal_idx, context_idx] = float(weight)
            if symmetric:
                matrix[context_idx, goal_idx] = float(weight)

        return cls(
            schema=schema,
            weights=matrix,
            influence_scale=influence_scale,
            min_factor=min_factor,
            max_factor=max_factor,
        )

    def factor_for(self, goal_idx: int, state: MotivationalState) -> float:
        if state.schema != self.schema:
            raise ValueError("state schema must match compatibility matrix schema")
        row = self.weights[goal_idx]
        normalizer = float(np.sum(np.abs(row)))
        if normalizer == 0.0:
            return 1.0

        influence = float(np.dot(row, state.G) / normalizer)
        return _clip_factor(
            1.0 + (self.influence_scale * influence),
            self.min_factor,
            self.max_factor,
        )
