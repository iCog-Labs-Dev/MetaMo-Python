from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from core.features import StimulusFeatures, GoalChangeFeedback
from core.schema import MotivationSchema
from core.state import MotivationalState


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class AppraisalProfile(ABC):
    """
    Configurable OpenPsi appraisal semantics.

    Core OpenPsi modulators are updated by profile-provided appraisal rules.
    Application-specific modulators can be updated by overriding.
    """

    name: str = "openpsi_base"
    required_modulators: tuple[str, ...] = (
        "valence",
        "arousal",
        "approach",
        "resolution",
        "threshold",
        "securing",
    )
    app_delta_scale: float = 1.0
    app_delta_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def validate(self, schema: MotivationSchema) -> None:
        missing = [name for name in self.required_modulators if name not in schema.modulator_names]
        if missing:
            raise ValueError(f"appraisal profile {self.name} missing modulators: {missing}")

    def stimulus_features(self, state: MotivationalState, stimulus: Any) -> StimulusFeatures:
        """
        Return already prepared stimulus features.
        """
        if isinstance(stimulus, StimulusFeatures):
            return stimulus
        raise TypeError(
            "No default MetaMo stimulus schema exists; application profiles must "
            "override stimulus_features() for raw stimulus objects"
        )

    @abstractmethod
    def core_modulator_deltas(
        self,
        state: MotivationalState,
        features: StimulusFeatures,
    ) -> dict[str, float]:
        """
        Convert profile-owned stimulus features into core OpenPsi modulator deltas.
        """
        pass

    def application_modulator_deltas(
        self,
        state: MotivationalState,
        features: StimulusFeatures,
    ) -> dict[str, float]:
        """
        Optional application specific modulator update rule.
        """
        return {}

    def goal_change_feedback(
        self,
        state: MotivationalState,
        stimulus: Any,
        features: StimulusFeatures,
    ) -> GoalChangeFeedback:
        """
        Produce optional feedback for the MAGUS goal-change calculator.
        """
        return GoalChangeFeedback()

    def bound_core_value(self, value: float) -> float:
        return float(sigmoid(4.0 * (value - 0.5)))

    def bound_application_value(self, name: str, value: float) -> float:
        lower, upper = self.app_delta_bounds.get(name, (0.0, 1.0))
        return float(np.clip(value, lower, upper))

    def appraise(self, state: MotivationalState, stimulus: Any) -> MotivationalState:
        self.validate(state.schema)
        extracted_features = self.stimulus_features(state, stimulus)
        next_state = state.copy()

        for name, delta in self.core_modulator_deltas(state, extracted_features).items():
            idx = next_state.modulator_index(name)
            next_state.M[idx] = self.bound_core_value(next_state.M[idx] + delta)

        for name, delta in self.application_modulator_deltas(state, extracted_features).items():
            if name not in state.schema.modulators.application_specific:
                raise ValueError(f"{name} is not an application-specific modulator")
            idx = next_state.modulator_index(name)
            next_state.M[idx] = self.bound_application_value(
                name,
                next_state.M[idx] + (self.app_delta_scale * delta),
            )

        return next_state
