from dataclasses import dataclass

from core.state import MotivationalState


@dataclass(frozen=True)
class StateLawCheckResult:
    """
    Numeric result for an approximate MetaMo law check.
    """

    principle: str
    left_state: MotivationalState
    right_state: MotivationalState
    error: float
    tolerance: float
    holds: bool
