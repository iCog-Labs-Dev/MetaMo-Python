from dataclasses import dataclass
import math


def clamp_unit(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("research assistant stimulus values must be finite")
    return max(0.0, min(1.0, numeric))


@dataclass(frozen=True)
class ResearchAssistantStimulus:
    """
    Raw application stimulus for the research-assistant application.

    These four fields are a research-assistant convention, not a MetaMo-wide
    stimulus schema.
    """

    novelty: float
    conduciveness: float
    risk: float
    effort: float

    def __post_init__(self):
        object.__setattr__(self, "novelty", clamp_unit(self.novelty))
        object.__setattr__(self, "conduciveness", clamp_unit(self.conduciveness))
        object.__setattr__(self, "risk", clamp_unit(self.risk))
        object.__setattr__(self, "effort", clamp_unit(self.effort))


@dataclass(frozen=True)
class ResearchAssistantSignals:
    """
    Application-owned semantic signals extracted from user text.

    These signals are not MetaMo-wide stimulus fields. They are used by the
    Research Assistant profile to build candidate actions and anti-goal
    feedback deterministically.
    """

    task_intent: str
    ambiguity: float
    citation_need: float
    comparison_need: float
    summary_need: float
    exploration_need: float
    unsafe_pressure: float
    privacy_pressure: float
    unsupported_claim_pressure: float
    context_loss_pressure: float

    def __post_init__(self):
        object.__setattr__(self, "task_intent", self.task_intent.strip().lower())
        for field_name in (
            "ambiguity",
            "citation_need",
            "comparison_need",
            "summary_need",
            "exploration_need",
            "unsafe_pressure",
            "privacy_pressure",
            "unsupported_claim_pressure",
            "context_loss_pressure",
        ):
            object.__setattr__(self, field_name, clamp_unit(getattr(self, field_name)))


@dataclass(frozen=True)
class ResearchAssistantPerception:
    """
    Full application perception returned by the Research Assistant LLM boundary.
    """

    stimulus: ResearchAssistantStimulus
    signals: ResearchAssistantSignals
