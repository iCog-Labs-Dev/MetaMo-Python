from dataclasses import dataclass

from applications.research_assistant.prompts import TASK_INTENTS
from applications.research_assistant.stimulus import (
    ResearchAssistantPerception,
    ResearchAssistantSignals,
    ResearchAssistantStimulus,
    clamp_unit,
)


EPSILON = 1e-9


@dataclass(frozen=True)
class CalibrationResult:
    """
    Result of deterministic Research Assistant perception calibration.

    Calibration does not replace the LLM and does not invent a missing
    perception. It only checks a complete perception object for internal
    consistency before the deterministic planner consumes it.
    """

    perception: ResearchAssistantPerception
    adjustments: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.adjustments)


def _floor(name: str, value: float, floor: float, notes: list[str]) -> float:
    calibrated = max(value, clamp_unit(floor))
    if calibrated > value + EPSILON:
        notes.append(f"{name} raised from {value:.3f} to {calibrated:.3f}")
    return calibrated


def _ceiling(name: str, value: float, ceiling: float, notes: list[str]) -> float:
    calibrated = min(value, clamp_unit(ceiling))
    if calibrated < value - EPSILON:
        notes.append(f"{name} lowered from {value:.3f} to {calibrated:.3f}")
    return calibrated


def _validate_task_intent(task_intent: str) -> None:
    if task_intent not in TASK_INTENTS:
        raise ValueError(f"unknown research assistant task intent: {task_intent}")


def calibrate_perception(
    perception: ResearchAssistantPerception,
) -> CalibrationResult:
    """
    Apply application-owned consistency rules to LLM semantic perception.

    The rules are intentionally conservative:

    - unsafe or privacy pressure must be reflected in stimulus risk;
    - unsupported-claim pressure must raise citation need and some risk;
    - ambiguity must raise context-loss pressure and some effort;
    - high safety pressure reduces exploration pressure.
    """
    stimulus = perception.stimulus
    signals = perception.signals
    _validate_task_intent(signals.task_intent)

    notes: list[str] = []
    unsafe = signals.unsafe_pressure
    privacy = signals.privacy_pressure
    unsupported = signals.unsupported_claim_pressure
    context_loss = signals.context_loss_pressure
    ambiguity = signals.ambiguity

    calibrated_context_loss = _floor(
        "context_loss_pressure",
        context_loss,
        0.70 * ambiguity,
        notes,
    )
    calibrated_citation = _floor(
        "citation_need",
        signals.citation_need,
        0.85 * unsupported,
        notes,
    )

    safety_pressure = max(unsafe, privacy)
    calibrated_exploration = signals.exploration_need
    if safety_pressure > 0.55:
        calibrated_exploration = _ceiling(
            "exploration_need",
            calibrated_exploration,
            1.0 - (0.70 * safety_pressure),
            notes,
        )

    calibrated_risk = _floor(
        "risk",
        stimulus.risk,
        max(
            0.90 * unsafe,
            0.85 * privacy,
            0.60 * unsupported,
            0.40 * calibrated_context_loss,
        ),
        notes,
    )
    calibrated_effort = _floor(
        "effort",
        stimulus.effort,
        max(
            0.65 * ambiguity,
            0.55 * calibrated_context_loss,
            0.25 * calibrated_citation,
        ),
        notes,
    )

    calibrated_stimulus = ResearchAssistantStimulus(
        novelty=stimulus.novelty,
        conduciveness=stimulus.conduciveness,
        risk=calibrated_risk,
        effort=calibrated_effort,
    )
    calibrated_signals = ResearchAssistantSignals(
        task_intent=signals.task_intent,
        ambiguity=signals.ambiguity,
        citation_need=calibrated_citation,
        comparison_need=signals.comparison_need,
        summary_need=signals.summary_need,
        exploration_need=calibrated_exploration,
        unsafe_pressure=signals.unsafe_pressure,
        privacy_pressure=signals.privacy_pressure,
        unsupported_claim_pressure=signals.unsupported_claim_pressure,
        context_loss_pressure=calibrated_context_loss,
    )

    return CalibrationResult(
        perception=ResearchAssistantPerception(
            stimulus=calibrated_stimulus,
            signals=calibrated_signals,
        ),
        adjustments=tuple(notes),
    )
