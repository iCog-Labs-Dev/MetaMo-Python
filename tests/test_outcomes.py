import unittest
import numpy as np

from core.outcomes import ActionObservation, ActionPrediction
from core.state import Action, MotivationalState
from core.schema import DEFAULT_MOTIVATION_SCHEMA


class TestActionPrediction(unittest.TestCase):

    def test_creates_valid_prediction(self):
        prediction = ActionPrediction(
            execution_id="execution-001",
            action_id="move_right",
            expected_effects={
                "energy_change": -2.0,
                "minerals_collected": 1.0,
            },
            context_ids=("gridworld-episode-01",),
        )

        self.assertEqual(prediction.execution_id, "execution-001")
        self.assertEqual(prediction.action_id, "move_right")
        self.assertEqual(prediction.expected_effects["energy_change"], -2.0)
        self.assertEqual(prediction.context_ids, ("gridworld-episode-01",))

    def test_rejects_empty_identifiers(self):
        with self.assertRaises(ValueError):
            ActionPrediction("", "move_right", {})

        with self.assertRaises(ValueError):
            ActionPrediction("execution-001", " ", {})

    def test_rejects_invalid_metric_names(self):
        with self.assertRaises(ValueError):
            ActionPrediction("execution-001", "move_right", {" ": 1.0})

    def test_rejects_non_finite_and_boolean_values(self):
        for invalid_value in (float("nan"), float("inf"), float("-inf"), True):
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValueError):
                    ActionPrediction(
                        "execution-001",
                        "move_right",
                        {"energy_change": invalid_value},
                    )

    def test_rejects_non_numeric_values(self):
        with self.assertRaises(ValueError):
            ActionPrediction(
                "execution-001",
                "move_right",
                {"energy_change": "unknown"},
            )

    def test_rejects_invalid_context_identifiers(self):
        with self.assertRaises(ValueError):
            ActionPrediction(
                "execution-001",
                "move_right",
                {},
                context_ids=("",),
            )

    def test_prediction_is_immutable(self):
        prediction = ActionPrediction(
            "execution-001",
            "move_right",
            {"energy_change": -2.0},
        )

        with self.assertRaises(TypeError):
            prediction.expected_effects["energy_change"] = 0.0

        with self.assertRaises(AttributeError):
            prediction.action_id = "move_left"


class TestActionObservation(unittest.TestCase):

    def test_creates_valid_observation(self):
        observation = ActionObservation(
            execution_id="execution-001",
            action_id="move_right",
            observed_effects={
                "energy_change": -2.0,
                "minerals_collected": 0.0,
            },
            context_ids=("gridworld-episode-01",),
            provenance={
                "energy_change": "gridworld_environment",
                "minerals_collected": "gridworld_environment",
            },
        )

        self.assertEqual(observation.execution_id, "execution-001")
        self.assertEqual(observation.action_id, "move_right")
        self.assertEqual(observation.observed_effects["energy_change"], -2.0)
        self.assertEqual(observation.observed_effects["minerals_collected"], 0.0)
        self.assertEqual(
            observation.provenance["energy_change"],
            "gridworld_environment",
        )

    def test_distinguishes_missing_measurement_from_zero(self):
        missing = ActionObservation(
            "execution-001",
            "move_right",
            {},
        )
        measured_zero = ActionObservation(
            "execution-002",
            "move_right",
            {"energy_change": 0.0},
            provenance={"energy_change": "gridworld_environment"},
        )

        self.assertNotIn("energy_change", missing.observed_effects)
        self.assertEqual(measured_zero.observed_effects["energy_change"], 0.0)


    def test_rejects_invalid_observation_identifiers_and_metrics(self):
        invalid_cases = (
            ("", "move_right", {"energy_change": 1.0}),
            ("execution-001", " ", {"energy_change": 1.0}),
            ("execution-001", "move_right", {" ": 1.0}),
            ("execution-001", "move_right", {"energy_change": "unknown"}),
        )

        for execution_id, action_id, effects in invalid_cases:
            with self.subTest(
                execution_id=execution_id,
                action_id=action_id,
                effects=effects,
            ):
                with self.assertRaises(ValueError):
                    ActionObservation(
                        execution_id,
                        action_id,
                        effects,
                        provenance={
                            metric: "environment"
                            for metric in effects
                        },
                    )


    def test_rejects_observation_without_provenance(self):
        with self.assertRaises(ValueError):
            ActionObservation(
                "execution-001",
                "move_right",
                {"energy_change": -2.0},
            )

    def test_rejects_incomplete_provenance(self):
        with self.assertRaises(ValueError):
            ActionObservation(
                "execution-001",
                "move_right",
                {"energy_change": -2.0, "minerals_collected": 1.0},
                provenance={"energy_change": "gridworld_environment"},
            )

    def test_rejects_extra_provenance(self):
        with self.assertRaises(ValueError):
            ActionObservation(
                "execution-001",
                "move_right",
                {"energy_change": -2.0},
                provenance={
                    "energy_change": "gridworld_environment",
                    "minerals_collected": "gridworld_environment",
                },
            )

    def test_rejects_invalid_provenance_source(self):
        with self.assertRaises(ValueError):
            ActionObservation(
                "execution-001",
                "move_right",
                {"energy_change": -2.0},
                provenance={"energy_change": " "},
            )

    def test_rejects_non_finite_and_boolean_values(self):
        for invalid_value in (float("nan"), float("inf"), float("-inf"), False):
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValueError):
                    ActionObservation(
                        "execution-001",
                        "move_right",
                        {"energy_change": invalid_value},
                        provenance={"energy_change": "environment"},
                    )

    def test_observation_is_immutable(self):
        observation = ActionObservation(
            "execution-001",
            "move_right",
            {"energy_change": -2.0},
            provenance={"energy_change": "environment"},
        )

        with self.assertRaises(TypeError):
            observation.observed_effects["energy_change"] = 0.0

        with self.assertRaises(TypeError):
            observation.provenance["energy_change"] = "another_source"

        with self.assertRaises(AttributeError):
            observation.action_id = "move_left"


class TestApplicationExamples(unittest.TestCase):

    def test_gridworld_can_use_shared_outcome_records(self):
        prediction = ActionPrediction(
            execution_id="gridworld-run-001",
            action_id="move_right",
            expected_effects={
                "energy_change": -2.0,
                "minerals_collected": 1.0,
            },
            context_ids=("gridworld-episode-01",),
        )

        observation = ActionObservation(
            execution_id="gridworld-run-001",
            action_id="move_right",
            observed_effects={
                "energy_change": -2.0,
                "minerals_collected": 0.0,
            },
            context_ids=("gridworld-episode-01",),
            provenance={
                "energy_change": "gridworld_environment",
                "minerals_collected": "gridworld_environment",
            },
        )

        self.assertEqual(prediction.execution_id, observation.execution_id)
        self.assertEqual(prediction.action_id, observation.action_id)

    def test_research_assistant_can_use_shared_outcome_records(self):
        prediction = ActionPrediction(
            execution_id="research-run-001",
            action_id="summarize_source",
            expected_effects={
                "response_generated": 1.0,
                "citation_count": 2.0,
            },
            context_ids=("research-task-01",),
        )

        observation = ActionObservation(
            execution_id="research-run-001",
            action_id="summarize_source",
            observed_effects={
                "response_generated": 1.0,
                "citation_count": 2.0,
                "unsupported_claim_count": 0.0,
            },
            context_ids=("research-task-01",),
            provenance={
                "response_generated": "response_generator",
                "citation_count": "citation_evaluator",
                "unsupported_claim_count": "claim_evaluator",
            },
        )

        self.assertEqual(prediction.execution_id, observation.execution_id)
        self.assertEqual(prediction.action_id, observation.action_id)
        self.assertEqual(
            observation.observed_effects["unsupported_claim_count"],
            0.0,
        )


class TestOutcomeCompatibility(unittest.TestCase):

    def test_prediction_defensively_copies_input_mapping(self):
        effects = {"energy_change": -2.0}

        prediction = ActionPrediction(
            "execution-001",
            "move_right",
            effects,
        )

        effects["energy_change"] = 10.0
        effects["new_metric"] = 5.0

        self.assertEqual(prediction.expected_effects["energy_change"], -2.0)
        self.assertNotIn("new_metric", prediction.expected_effects)

    def test_observation_defensively_copies_input_mappings(self):
        effects = {"energy_change": -2.0}
        provenance = {"energy_change": "gridworld_environment"}

        observation = ActionObservation(
            "execution-001",
            "move_right",
            effects,
            provenance=provenance,
        )

        effects["energy_change"] = 10.0
        provenance["energy_change"] = "modified_source"

        self.assertEqual(observation.observed_effects["energy_change"], -2.0)
        self.assertEqual(
            observation.provenance["energy_change"],
            "gridworld_environment",
        )

    def test_existing_action_and_motivational_state_apis_remain_usable(self):
        schema = DEFAULT_MOTIVATION_SCHEMA

        action = Action(
            id="move_right",
            goal_correlations=np.zeros(schema.num_goals),
            delta_g=np.zeros(schema.num_goals),
            schema=schema,
        )

        state = MotivationalState(
            G=np.zeros(schema.num_goals),
            M=np.zeros(schema.num_modulators),
            schema=schema,
        )

        state_copy = state.copy()

        self.assertEqual(action.id, "move_right")
        self.assertEqual(action.goal_correlations.shape[0], schema.num_goals)
        self.assertEqual(action.delta_g.shape[0], schema.num_goals)
        self.assertEqual(state.G.shape[0], schema.num_goals)
        self.assertEqual(state.M.shape[0], schema.num_modulators)
        self.assertIsNot(state, state_copy)
        np.testing.assert_array_equal(state.G, state_copy.G)
        np.testing.assert_array_equal(state.M, state_copy.M)


if __name__ == "__main__":
    unittest.main()
