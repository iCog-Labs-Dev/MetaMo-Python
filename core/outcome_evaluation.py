from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping

from core.outcomes import ActionObservation, ActionPrediction


class ObservationStatus(Enum):
    """Describes how completely the expected effects were observed."""

    PENDING = "pending"
    PARTIALLY_OBSERVED = "partially_observed"
    FULLY_OBSERVED = "fully_observed"


class SuccessStatus(Enum):
    """Describes whether application-defined task success was evaluated."""

    NOT_EVALUATED = "not_evaluated"
    EVALUATED = "evaluated"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class OutcomeEvaluation:
    """Stores the result of comparing an action prediction with its observation."""

    execution_id: str
    action_id: str
    prediction_errors: Mapping[str, float]
    missing_metrics: tuple[str, ...]
    unexpected_metrics: tuple[str, ...]
    observation_status: ObservationStatus
    task_success: bool | None
    success_status: SuccessStatus

    def __post_init__(self) -> None:
        # Copy collections so the result cannot be changed through external references.
        object.__setattr__(
            self,
            "prediction_errors",
            MappingProxyType(dict(self.prediction_errors)),
        )
        object.__setattr__(self, "missing_metrics", tuple(self.missing_metrics))
        object.__setattr__(
            self,
            "unexpected_metrics",
            tuple(self.unexpected_metrics),
        )


def evaluate_outcome(
    prediction: ActionPrediction,
    observation: ActionObservation,
    success_criteria: Callable[[Mapping[str, float]], bool | None] | None = None,
) -> OutcomeEvaluation:
    """Compare predicted and observed effects without modifying either record.

    Calculates prediction errors as observed minus expected, and reports
    missing and unexpected metrics separately. Observation completeness
    does not determine task success.

    Optional success criteria must be deterministic and side-effect-free,
    and return exactly True, False, or None. Without criteria, task
    success remains unknown.
    """

    if prediction.execution_id != observation.execution_id:
        raise ValueError("prediction and observation execution IDs do not match")

    if prediction.action_id != observation.action_id:
        raise ValueError("prediction and observation action IDs do not match")

    expected = prediction.expected_effects
    observed = observation.observed_effects

    matching_metrics = expected.keys() & observed.keys()

    prediction_errors = {
        metric: observed[metric] - expected[metric]
        for metric in sorted(matching_metrics)
    }

    missing_metrics = tuple(sorted(expected.keys() - observed.keys()))
    unexpected_metrics = tuple(sorted(observed.keys() - expected.keys()))

    # An empty prediction gives us no expected metrics to measure completeness against.
    if not expected or not matching_metrics:
        observation_status = ObservationStatus.PENDING
    elif missing_metrics:
        observation_status = ObservationStatus.PARTIALLY_OBSERVED
    else:
        observation_status = ObservationStatus.FULLY_OBSERVED

    if success_criteria is None:
        task_success = None
        success_status = SuccessStatus.NOT_EVALUATED
    else:
        task_success = success_criteria(observed)

        if (
            task_success is not True
            and task_success is not False
            and task_success is not None
        ):
            raise ValueError(
                "success criteria must return exactly True, False, or None"
            )

        success_status = (
            SuccessStatus.INDETERMINATE
            if task_success is None
            else SuccessStatus.EVALUATED
        )

    return OutcomeEvaluation(
        execution_id=prediction.execution_id,
        action_id=prediction.action_id,
        prediction_errors=prediction_errors,
        missing_metrics=missing_metrics,
        unexpected_metrics=unexpected_metrics,
        observation_status=observation_status,
        task_success=task_success,
        success_status=success_status,
    )
