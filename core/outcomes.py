from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping


def _validate_identifier(value: str, field_name: str) -> None:
    """Ensure an identifier is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_context_ids(context_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and normalize context identifiers."""
    if not isinstance(context_ids, (tuple, list)):
        raise ValueError("context_ids must be a tuple or list of strings")

    validated = tuple(context_ids)

    for context_id in validated:
        _validate_identifier(context_id, "context_id")

    return validated


def _validate_effects(
    effects: Mapping[str, float],
    field_name: str,
) -> Mapping[str, float]:
    """Validate metric names and finite numeric values."""
    if not isinstance(effects, Mapping):
        raise ValueError(f"{field_name} must be a mapping")

    validated = {}

    for metric_name, value in effects.items():
        _validate_identifier(metric_name, "metric name")

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name}[{metric_name!r}] must be numeric")

        numeric_value = float(value)

        if not isfinite(numeric_value):
            raise ValueError(f"{field_name}[{metric_name!r}] must be finite")

        validated[metric_name] = numeric_value

    return MappingProxyType(validated)


@dataclass(frozen=True)
class ActionPrediction:
    """
    Describes the expected effects of one particular action execution.

    These application-level effects are separate from Action.delta_g,
    which represents expected changes to motivational goal weights.
    """

    execution_id: str
    action_id: str
    expected_effects: Mapping[str, float]
    context_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.execution_id, "execution_id")
        _validate_identifier(self.action_id, "action_id")

        object.__setattr__(
            self,
            "expected_effects",
            _validate_effects(self.expected_effects, "expected_effects"),
        )
        object.__setattr__(
            self,
            "context_ids",
            _validate_context_ids(self.context_ids),
        )


@dataclass(frozen=True)
class ActionObservation:
    """
    Describes the measured effects of one particular action execution.

    An empty observed_effects mapping means no measurements were recorded.
    A metric with value 0.0 means that a zero change was actually measured.

    These application-level effects are separate from Action.delta_g,
    which represents expected changes to motivational goal weights.
    """

    execution_id: str
    action_id: str
    observed_effects: Mapping[str, float]
    context_ids: tuple[str, ...] = ()
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.execution_id, "execution_id")
        _validate_identifier(self.action_id, "action_id")

        observed = _validate_effects(self.observed_effects, "observed_effects")

        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")

        validated_provenance = {}

        for metric_name, source in self.provenance.items():
            _validate_identifier(metric_name, "provenance metric name")
            _validate_identifier(source, "provenance source")
            validated_provenance[metric_name] = source

        if set(observed) != set(validated_provenance):
            raise ValueError(
                "provenance keys must match observed_effects keys"
            )

        object.__setattr__(self, "observed_effects", observed)
        object.__setattr__(
            self,
            "context_ids",
            _validate_context_ids(self.context_ids),
        )
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(validated_provenance),
        )
