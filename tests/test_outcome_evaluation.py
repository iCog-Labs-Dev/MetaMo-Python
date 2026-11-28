import unittest

from core.outcomes import ActionObservation, ActionPrediction
from core.outcome_evaluation import ObservationStatus, SuccessStatus, evaluate_outcome


def make_prediction(expected_effects=None, execution_id="execution-001", action_id="move_right"):
    if expected_effects is None:
        expected_effects = {}

    return ActionPrediction(
        execution_id=execution_id,
        action_id=action_id,
        expected_effects=expected_effects,
    )


def make_observation(observed_effects=None, execution_id="execution-001", action_id="move_right"):
    if observed_effects is None:
        observed_effects = {}

    return ActionObservation(
        execution_id=execution_id,
        action_id=action_id,
        observed_effects=observed_effects,
        provenance={
            metric: "test_environment"
            for metric in observed_effects
        },
    )


class TestOutcomeEvaluation(unittest.TestCase):

    def test_calculates_prediction_errors(self):
        prediction = make_prediction({
            "energy_change": -2.0,
            "minerals_collected": 1.0,
        })
        observation = make_observation({
            "energy_change": -2.0,
            "minerals_collected": 0.0,
        })

        result = evaluate_outcome(prediction, observation)

        self.assertEqual(
            dict(result.prediction_errors),
            {
                "energy_change": 0.0,
                "minerals_collected": -1.0,
            },
        )

    def test_reports_missing_metrics_without_treating_them_as_zero(self):
        prediction = make_prediction({
            "energy_change": -2.0,
            "minerals_collected": 1.0,
        })
        observation = make_observation({
            "energy_change": -2.0,
        })

        result = evaluate_outcome(prediction, observation)

        self.assertEqual(result.missing_metrics, ("minerals_collected",))
        self.assertNotIn("minerals_collected", result.prediction_errors)

    def test_measured_zero_is_a_valid_observation(self):
        prediction = make_prediction({"minerals_collected": 1.0})
        observation = make_observation({"minerals_collected": 0.0})

        result = evaluate_outcome(prediction, observation)

        self.assertEqual(result.prediction_errors["minerals_collected"], -1.0)
        self.assertEqual(result.missing_metrics, ())

    def test_reports_unexpected_metrics_separately(self):
        prediction = make_prediction({"response_generated": 1.0})
        observation = make_observation({
            "response_generated": 1.0,
            "unsupported_claim_count": 0.0,
        })

        result = evaluate_outcome(prediction, observation)

        self.assertEqual(
            result.unexpected_metrics,
            ("unsupported_claim_count",),
        )
        self.assertNotIn("unsupported_claim_count", result.prediction_errors)

    def test_rejects_mismatched_execution_ids(self):
        prediction = make_prediction(execution_id="execution-001")
        observation = make_observation(execution_id="execution-002")

        with self.assertRaises(ValueError):
            evaluate_outcome(prediction, observation)

    def test_rejects_mismatched_action_ids(self):
        prediction = make_prediction(action_id="move_right")
        observation = make_observation(action_id="move_left")

        with self.assertRaises(ValueError):
            evaluate_outcome(prediction, observation)


class TestObservationStatus(unittest.TestCase):

    def setUp(self):
        self.prediction = make_prediction({
            "energy_change": -2.0,
            "minerals_collected": 1.0,
        })

    def test_status_is_pending_when_no_expected_metrics_are_observed(self):
        observation = make_observation({})

        result = evaluate_outcome(self.prediction, observation)

        self.assertEqual(result.observation_status, ObservationStatus.PENDING)

    def test_status_is_partially_observed_when_some_metrics_are_missing(self):
        observation = make_observation({
            "energy_change": -2.0,
        })

        result = evaluate_outcome(self.prediction, observation)

        self.assertEqual(
            result.observation_status,
            ObservationStatus.PARTIALLY_OBSERVED,
        )

    def test_status_is_fully_observed_when_all_expected_metrics_are_present(self):
        observation = make_observation({
            "energy_change": -2.0,
            "minerals_collected": 0.0,
        })

        result = evaluate_outcome(self.prediction, observation)

        self.assertEqual(
            result.observation_status,
            ObservationStatus.FULLY_OBSERVED,
        )

    def test_unexpected_metrics_do_not_count_toward_completeness(self):
        observation = make_observation({
            "unsupported_claim_count": 0.0,
        })

        result = evaluate_outcome(self.prediction, observation)

        self.assertEqual(result.observation_status, ObservationStatus.PENDING)
        self.assertEqual(
            result.unexpected_metrics,
            ("unsupported_claim_count",),
        )


class TestSuccessCriteria(unittest.TestCase):

    def setUp(self):
        self.prediction = make_prediction({"energy_change": -2.0})
        self.observation = make_observation({"energy_change": -2.0})

    def test_success_is_not_evaluated_without_criteria(self):
        result = evaluate_outcome(self.prediction, self.observation)

        self.assertIsNone(result.task_success)
        self.assertEqual(result.success_status, SuccessStatus.NOT_EVALUATED)

    def test_success_is_true_when_criteria_return_true(self):
        result = evaluate_outcome(
            self.prediction,
            self.observation,
            success_criteria=lambda observed: observed["energy_change"] == -2.0,
        )

        self.assertIs(result.task_success, True)
        self.assertEqual(result.success_status, SuccessStatus.EVALUATED)

    def test_success_is_false_when_criteria_return_false(self):
        result = evaluate_outcome(
            self.prediction,
            self.observation,
            success_criteria=lambda observed: observed["energy_change"] == 0.0,
        )

        self.assertIs(result.task_success, False)
        self.assertEqual(result.success_status, SuccessStatus.EVALUATED)

    def test_success_is_indeterminate_when_criteria_return_none(self):
        result = evaluate_outcome(
            self.prediction,
            self.observation,
            success_criteria=lambda observed: None,
        )

        self.assertIsNone(result.task_success)
        self.assertEqual(result.success_status, SuccessStatus.INDETERMINATE)


class TestInvalidSuccessCriteria(unittest.TestCase):

    def setUp(self):
        self.prediction = make_prediction({"energy_change": -2.0})
        self.observation = make_observation({"energy_change": -2.0})

    def test_rejects_integer_one_as_success_result(self):
        with self.assertRaises(ValueError):
            evaluate_outcome(
                self.prediction,
                self.observation,
                success_criteria=lambda observed: 1,
            )

    def test_rejects_string_as_success_result(self):
        with self.assertRaises(ValueError):
            evaluate_outcome(
                self.prediction,
                self.observation,
                success_criteria=lambda observed: "yes",
            )

    def test_rejects_list_as_success_result(self):
        with self.assertRaises(ValueError):
            evaluate_outcome(
                self.prediction,
                self.observation,
                success_criteria=lambda observed: [],
            )


class TestSuccessCriteriaBehavior(unittest.TestCase):

    def setUp(self):
        self.prediction = make_prediction({"energy_change": -2.0})
        self.observation = make_observation({"energy_change": -2.0})

    def test_success_criteria_receive_read_only_observations(self):
        def criteria(observed):
            observed["energy_change"] = 0.0
            return True

        with self.assertRaises(TypeError):
            evaluate_outcome(
                self.prediction,
                self.observation,
                success_criteria=criteria,
            )

    def test_success_criteria_exceptions_propagate(self):
        def criteria(observed):
            raise RuntimeError("criteria failed")

        with self.assertRaisesRegex(RuntimeError, "criteria failed"):
            evaluate_outcome(
                self.prediction,
                self.observation,
                success_criteria=criteria,
            )

    def test_criteria_can_return_none_when_required_metric_is_missing(self):
        prediction = make_prediction({"energy_change": -2.0})
        observation = make_observation({})

        def criteria(observed):
            if "energy_change" not in observed:
                return None
            return observed["energy_change"] >= -2.0

        result = evaluate_outcome(
            prediction,
            observation,
            success_criteria=criteria,
        )

        self.assertIsNone(result.task_success)
        self.assertEqual(result.success_status, SuccessStatus.INDETERMINATE)


class TestEmptyPredictions(unittest.TestCase):

    def test_empty_prediction_with_empty_observation_is_pending(self):
        prediction = make_prediction({})
        observation = make_observation({})

        result = evaluate_outcome(prediction, observation)

        self.assertEqual(result.observation_status, ObservationStatus.PENDING)
        self.assertEqual(dict(result.prediction_errors), {})
        self.assertEqual(result.missing_metrics, ())
        self.assertEqual(result.unexpected_metrics, ())

    def test_empty_prediction_reports_all_observations_as_unexpected(self):
        prediction = make_prediction({})
        observation = make_observation({
            "energy_change": -2.0,
            "minerals_collected": 1.0,
        })

        result = evaluate_outcome(prediction, observation)

        self.assertEqual(result.observation_status, ObservationStatus.PENDING)
        self.assertEqual(
            result.unexpected_metrics,
            ("energy_change", "minerals_collected"),
        )
        self.assertEqual(dict(result.prediction_errors), {})

    def test_success_criteria_can_evaluate_empty_prediction(self):
        prediction = make_prediction({})
        observation = make_observation({"response_generated": 1.0})

        result = evaluate_outcome(
            prediction,
            observation,
            success_criteria=lambda observed: observed["response_generated"] == 1.0,
        )

        self.assertEqual(result.observation_status, ObservationStatus.PENDING)
        self.assertIs(result.task_success, True)
        self.assertEqual(result.success_status, SuccessStatus.EVALUATED)


class TestEvaluationImmutability(unittest.TestCase):

    def test_evaluation_does_not_modify_input_records(self):
        expected = {"energy_change": -2.0}
        observed = {"energy_change": -1.0}

        prediction = make_prediction(expected)
        observation = make_observation(observed)

        original_expected = dict(prediction.expected_effects)
        original_observed = dict(observation.observed_effects)

        evaluate_outcome(prediction, observation)

        self.assertEqual(dict(prediction.expected_effects), original_expected)
        self.assertEqual(dict(observation.observed_effects), original_observed)

    def test_prediction_errors_mapping_is_read_only(self):
        prediction = make_prediction({"energy_change": -2.0})
        observation = make_observation({"energy_change": -1.0})

        result = evaluate_outcome(prediction, observation)

        with self.assertRaises(TypeError):
            result.prediction_errors["energy_change"] = 100.0

    def test_result_attributes_cannot_be_reassigned(self):
        prediction = make_prediction({"energy_change": -2.0})
        observation = make_observation({"energy_change": -1.0})

        result = evaluate_outcome(prediction, observation)

        with self.assertRaises(AttributeError):
            result.task_success = True

    def test_missing_and_unexpected_metrics_are_immutable_tuples(self):
        prediction = make_prediction({
            "energy_change": -2.0,
            "minerals_collected": 1.0,
        })
        observation = make_observation({
            "unsupported_claim_count": 0.0,
        })

        result = evaluate_outcome(prediction, observation)

        self.assertIsInstance(result.missing_metrics, tuple)
        self.assertIsInstance(result.unexpected_metrics, tuple)


class TestEvaluationDeterminism(unittest.TestCase):

    def test_same_inputs_produce_same_evaluation(self):
        prediction = make_prediction({
            "energy_change": -2.0,
            "minerals_collected": 1.0,
        })
        observation = make_observation({
            "energy_change": -1.0,
            "minerals_collected": 0.0,
        })

        first = evaluate_outcome(prediction, observation)
        second = evaluate_outcome(prediction, observation)

        self.assertEqual(first, second)

    def test_evaluation_preserves_execution_and_action_ids(self):
        prediction = make_prediction(
            {"energy_change": -2.0},
            execution_id="execution-123",
            action_id="collect_minerals",
        )
        observation = make_observation(
            {"energy_change": -1.0},
            execution_id="execution-123",
            action_id="collect_minerals",
        )

        result = evaluate_outcome(prediction, observation)

        self.assertEqual(result.execution_id, "execution-123")
        self.assertEqual(result.action_id, "collect_minerals")



class TestSuccessCriteriaUnexpectedMetrics(unittest.TestCase):

    def test_success_criteria_can_access_unexpected_metrics(self):
        prediction = make_prediction({"response_generated": 1.0})
        observation = make_observation({
            "response_generated": 1.0,
            "unsupported_claim_count": 0.0,
        })

        result = evaluate_outcome(
            prediction,
            observation,
            success_criteria=lambda observed: (
                observed["unsupported_claim_count"] == 0.0
            ),
        )

        self.assertIs(result.task_success, True)
        self.assertEqual(result.success_status, SuccessStatus.EVALUATED)


if __name__ == "__main__":
    unittest.main()
