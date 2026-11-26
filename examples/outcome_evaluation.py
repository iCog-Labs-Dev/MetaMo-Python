"""Demonstrate how to evaluate predicted and observed action outcomes.

Run from the repository root with:
    python -m examples.outcome_evaluation
"""

from core.outcomes import ActionObservation, ActionPrediction
from core.outcome_evaluation import evaluate_outcome


prediction = ActionPrediction(
    execution_id="execution-001",
    action_id="move_right",
    expected_effects={
        "position_change": 1.0,
        "energy_change": -2.0,
    },
)

observation = ActionObservation(
    execution_id="execution-001",
    action_id="move_right",
    observed_effects={
        "position_change": 1.0,
        "energy_change": -3.0,
    },
    provenance={
        "position_change": "environment",
        "energy_change": "environment",
    },
)

result = evaluate_outcome(prediction, observation)

print("Prediction errors:", dict(result.prediction_errors))
print("Missing metrics:", result.missing_metrics)
print("Unexpected metrics:", result.unexpected_metrics)
print("Observation status:", result.observation_status.value)
print("Task success:", result.task_success)
print("Success status:", result.success_status.value)

# Evaluate the outcome using application-defined success criteria.
result_with_criteria = evaluate_outcome(
    prediction,
    observation,
    success_criteria=lambda observed: (
        observed["position_change"] >= 1.0
        and observed["energy_change"] >= -3.0
    ),
)

print("\nEvaluation with success criteria:")
print("Task success:", result_with_criteria.task_success)
print("Success status:", result_with_criteria.success_status.value)
