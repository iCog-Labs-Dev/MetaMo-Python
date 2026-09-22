import unittest

import numpy as np

from core.schema import MagusGoalSchema, ModulatorSchema, MotivationSchema
from core.state import MotivationalState
from dynamics.stability import is_in_safe_region, project_to_safe_region


SCHEMA = MotivationSchema(
    goals=MagusGoalSchema(
        primary_goals=tuple(f"goal_{idx}" for idx in range(8)),
        anti_goals=("harm_a", "harm_b"),
    ),
    modulators=ModulatorSchema(),
)


class StabilityPropertyTests(unittest.TestCase):
    def test_projection_is_safe_and_idempotent_for_many_states(self):
        rng = np.random.default_rng(17)
        for _ in range(200):
            state = MotivationalState(
                G=rng.random(SCHEMA.num_goals),
                M=rng.random(SCHEMA.num_modulators),
                schema=SCHEMA,
            )
            projected = project_to_safe_region(state)
            projected_twice = project_to_safe_region(projected)

            self.assertTrue(is_in_safe_region(projected))
            self.assertLess(projected.distance_to(projected_twice), 1e-10)


if __name__ == "__main__":
    unittest.main()
