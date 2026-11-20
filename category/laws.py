from dataclasses import dataclass
from enum import Enum

from core.state import MotivationalState


class LawMode(str, Enum):
    """How an approximate law participates in runtime execution."""

    DISABLED = "disabled"
    OBSERVE = "observe"
    ENFORCE = "enforce"


@dataclass(frozen=True)
class RuntimeLawPolicy:
    """Runtime treatment of approximate categorical laws.

    Safety projection remains a hard invariant in the stability policy.
    Approximate laws are observed by default and only alter state when a
    caller explicitly opts into ENFORCE.
    """

    lax_distributive: LawMode = LawMode.OBSERVE
    contractive: LawMode = LawMode.OBSERVE


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
    evaluated: bool = True
    left_action_id: str | None = None
    right_action_id: str | None = None
    action_holds: bool = True
