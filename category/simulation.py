from dataclasses import dataclass

from core.state import Action, MotivationalState


@dataclass(frozen=True)
class ReciprocalSimulationResult:
    """
    Result of the Principle 2 commuting-update check.
    """

    source_action: Action
    target_action: Action
    translated_after_source_update: MotivationalState
    target_after_translation_update: MotivationalState
    error: float
    tolerance: float
    holds: bool
