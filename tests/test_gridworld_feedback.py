import unittest

from applications.gridworld.environment import GridWorld
from applications.gridworld.profile import (
    GRIDWORLD_APPRAISAL_PROFILE,
    GRIDWORLD_DECISION_PROFILE,
    GRIDWORLD_PROFILE,
)
from applications.gridworld.stimulus import GridWorldStimulus
from magus.decision import MagusDecision
from openpsi.appraisal import OpenPsiAppraisal


class GridWorldFeedbackTests(unittest.TestCase):
    def test_hazard_feedback_updates_overgoals_and_anti_goals(self):
        state = GRIDWORLD_PROFILE.create_initial_state()
        stimulus = GridWorldStimulus(
            goal_proximity=0.2,
            hazard_pressure=1.0,
            energy_deficit=0.5,
            time_pressure=0.5,
            safe_progress=0.1,
            safe_mobility=0.25,
        )
        appraised, feedback = OpenPsiAppraisal(
            GRIDWORLD_APPRAISAL_PROFILE
        ).appraise_with_feedback(state, stimulus)
        candidate = GRIDWORLD_PROFILE.build_candidates(GridWorld(seed=1).reset())[0]
        delta = MagusDecision(GRIDWORLD_DECISION_PROFILE).propose_delta_g(
            appraised, candidate, feedback
        )

        self.assertGreater(delta[state.goal_index("individuation")], 0.0)
        self.assertLess(delta[state.goal_index("transcendence")], 0.0)
        self.assertGreater(delta[state.goal_index("lava_exposure")], 0.0)
        self.assertGreater(delta[state.goal_index("boundary_collision")], 0.0)


if __name__ == "__main__":
    unittest.main()
