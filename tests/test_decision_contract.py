import unittest

import numpy as np

from core.decision import DecisionResult
from core.schema import MagusGoalSchema, ModulatorSchema, MotivationSchema
from core.state import Action, MotivationalState
from magus.decision import MagusDecision
from magus.goal_change import GoalChangeCalculator
from magus.profile import DecisionProfile


SCHEMA = MotivationSchema(
    goals=MagusGoalSchema(primary_goals=("task",), anti_goals=("harm",)),
    modulators=ModulatorSchema(),
)


class CountingGoalChangeCalculator(GoalChangeCalculator):
    def __init__(self):
        self.calls = 0

    def delta_g(
        self,
        state,
        candidate,
        feedback=None,
        *,
        lambda_ind,
        lambda_trans,
    ):
        self.calls += 1
        return candidate.delta_g.copy()


def make_state():
    return MotivationalState(
        G=np.array([0.8, 0.4, 0.7, 0.5]),
        M=np.full(SCHEMA.num_modulators, 0.5),
        schema=SCHEMA,
    )


def make_action(action_id, *, task_delta=0.0, risk=0.2, opportunity=0.4):
    correlations = np.zeros(SCHEMA.num_goals)
    correlations[SCHEMA.goal_index("task")] = 0.5
    delta = np.zeros(SCHEMA.num_goals)
    delta[SCHEMA.goal_index("task")] = task_delta
    return Action(
        id=action_id,
        goal_correlations=correlations,
        delta_g=delta,
        schema=SCHEMA,
        metadata={"risk": risk, "opportunity": opportunity},
    )


class DecisionContractTests(unittest.TestCase):
    def test_decision_returns_evidence_and_reuses_computed_delta(self):
        calculator = CountingGoalChangeCalculator()
        decision = MagusDecision(
            DecisionProfile(
                name="test",
                schema=SCHEMA,
                expected_goal_change_weight=1.0,
            ),
            goal_change_calculator=calculator,
        )
        candidates = [
            make_action("negative", task_delta=-0.05),
            make_action("positive", task_delta=0.05),
        ]

        result = decision.decide(make_state(), candidates)

        self.assertIsInstance(result, DecisionResult)
        self.assertEqual(result.action.id, "positive")
        self.assertEqual(len(result.candidate_scores), 2)
        self.assertEqual(calculator.calls, 2)
        self.assertGreater(result.chosen_score.expected_goal_change, 0.0)
        action, delta = result
        self.assertIs(action, result.action)
        np.testing.assert_array_equal(delta, result.proposed_delta_g)

    def test_overgoal_effects_are_candidate_dependent(self):
        calculator = CountingGoalChangeCalculator()
        decision = MagusDecision(
            DecisionProfile(name="test", schema=SCHEMA),
            goal_change_calculator=calculator,
        )
        risky = make_action("risky", risk=0.9, opportunity=0.1)
        safe = make_action("safe", risk=0.1, opportunity=0.6)

        result = decision.decide(make_state(), [risky, safe])
        scores = {item.action_id: item for item in result.candidate_scores}

        self.assertEqual(result.action.id, "safe")
        self.assertLess(
            scores["risky"].individuation_adjustment,
            scores["safe"].individuation_adjustment,
        )
        self.assertGreater(
            scores["safe"].transcendence_adjustment,
            scores["risky"].transcendence_adjustment,
        )


if __name__ == "__main__":
    unittest.main()
