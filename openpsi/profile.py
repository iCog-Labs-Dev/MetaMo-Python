from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from core.evidence import AppraisalEvidence
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

    def appraisal_evidence(self, state: MotivationalState, stimulus: Any) -> AppraisalEvidence:
        """
        Return already prepared appraisal evidence.
        """
        if isinstance(stimulus, AppraisalEvidence):
            return stimulus
        raise TypeError(
            "No default MetaMo stimulus schema exists; application profiles must "
            "override appraisal_evidence() for raw stimulus objects"
        )

    @abstractmethod
    def core_modulator_deltas(
        self,
        state: MotivationalState,
        evidence: AppraisalEvidence,
    ) -> dict[str, float]:
        """
        Convert profile-owned evidence into core OpenPsi modulator deltas.

        Applications implement this hook according to their own stimulus and
        evidence semantics.
        """
        pass

    def application_modulator_deltas(
        self,
        state: MotivationalState,
        evidence: AppraisalEvidence,
    ) -> dict[str, float]:
        """
        Optional application specific modulator update rule.
        """
        return {}

    def bound_core_value(self, value: float) -> float:
        return float(sigmoid(4.0 * (value - 0.5)))

    def bound_application_value(self, name: str, value: float) -> float:
        lower, upper = self.app_delta_bounds.get(name, (0.0, 1.0))
        return float(np.clip(value, lower, upper))

    def appraise(self, state: MotivationalState, stimulus: Any) -> MotivationalState:
        self.validate(state.schema)
        stimulus_evidence = self.appraisal_evidence(state, stimulus)
        next_state = state.copy()

        for name, delta in self.core_modulator_deltas(state, stimulus_evidence).items():
            idx = next_state.modulator_index(name)
            next_state.M[idx] = self.bound_core_value(next_state.M[idx] + delta)

        for name, delta in self.application_modulator_deltas(state, stimulus_evidence).items():
            if name not in state.schema.modulators.application_specific:
                raise ValueError(f"{name} is not an application-specific modulator")
            idx = next_state.modulator_index(name)
            next_state.M[idx] = self.bound_application_value(
                name,
                next_state.M[idx] + (self.app_delta_scale * delta),
            )

        return next_state
