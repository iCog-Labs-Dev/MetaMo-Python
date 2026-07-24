from dataclasses import dataclass
from typing import Any, Mapping


def _coerce_numeric(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


class _FeatureMap:
    container_label = "FeatureMap"

    features: Mapping[str, Any]

    def __init__(self, *args: Any, **kwargs: Any):
        features = kwargs.pop("features", None)
        merged = dict(features or {})

        if args:
            if len(args) == 1 and isinstance(args[0], Mapping):
                merged.update(args[0])
            else:
                raise TypeError(
                    f"{self.container_label} accepts a single mapping positional argument "
                    "or keyword feature values"
                )

        merged.update(kwargs)
        if any(not isinstance(name, str) or not name.strip() for name in merged):
            raise ValueError(f"{self.container_label} feature names must be non-empty strings")
        object.__setattr__(self, "features", dict(merged))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        return cls(values)

    def get(self, name: str, default: Any = 0.0) -> Any:
        if name in self.features:
            return self.features[name]
        return default

    def numeric(self, name: str, default: float = 0.0, clamp: bool = True) -> float:
        value = _coerce_numeric(self.get(name, default), name)
        return _clip_unit(value) if clamp else value

    def as_dict(self) -> dict[str, Any]:
        return dict(self.features)

    def as_tuple(self, names: tuple[str, ...]) -> tuple[Any, ...]:
        return tuple(self.get(name) for name in names)


@dataclass(frozen=True, init=False)
class StimulusFeatures(_FeatureMap):
    """
    Dynamic features produced by an application appraisal profile.
    """

    container_label = "StimulusFeatures"
    features: Mapping[str, Any]


@dataclass(frozen=True, init=False)
class GoalChangeFeedback(_FeatureMap):
    """
    Dynamic feedback consumed by a goal-change calculator.
    """

    container_label = "GoalChangeFeedback"
    features: Mapping[str, Any]
