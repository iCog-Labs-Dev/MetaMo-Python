from dataclasses import dataclass
from typing import Any, Mapping


def _coerce_numeric(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True, init=False)
class AppraisalEvidence:
    """
    Dynamic evidence produced by an application appraisal profile.
    """

    features: Mapping[str, Any]

    def __init__(self, *args: Any, **kwargs: Any):
        features = kwargs.pop("features", None)
        merged = dict(features or {})

        if args:
            if len(args) == 1 and isinstance(args[0], Mapping):
                merged.update(args[0])
            else:
                raise TypeError(
                    "AppraisalEvidence accepts a single mapping positional argument "
                    "or keyword feature values"
                )

        merged.update(kwargs)
        if any(not isinstance(name, str) or not name.strip() for name in merged):
            raise ValueError("appraisal evidence feature names must be non-empty strings")
        object.__setattr__(self, "features", dict(merged))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AppraisalEvidence":
        return cls(values)

    def get(self, name: str, default: Any = 0.0) -> Any:
        if name in self.features:
            return self.features[name]
        return default

    def numeric(self, name: str, default: float = 0.0, clamp: bool = True,) -> float:
        value = _coerce_numeric(self.get(name, default), name)
        return _clip_unit(value) if clamp else value

    def as_dict(self) -> dict[str, Any]:
        return dict(self.features)

    def as_tuple(self, names: tuple[str, ...]) -> tuple[Any, ...]:
        return tuple(self.get(name) for name in names)
