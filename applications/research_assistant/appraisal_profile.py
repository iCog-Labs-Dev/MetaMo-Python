from typing import Any

from applications.common.openpsi_conventions import OpenPsiFeatureMetrics
from applications.research_assistant.stimulus import (
    ResearchAssistantPerception,
    ResearchAssistantStimulus,
)
from core.features import GoalChangeFeedback, StimulusFeatures
from core.state import MotivationalState
from openpsi.profile import AppraisalProfile


class ResearchAssistantAppraisalProfile(OpenPsiFeatureMetrics, AppraisalProfile):
    """Stimulus feature mapping owned by the research-assistant application."""

    def stimulus_features(
        self,
        state: MotivationalState,
        stimulus: Any,
    ) -> StimulusFeatures:
        if isinstance(stimulus, StimulusFeatures):
            return stimulus
        if isinstance(stimulus, ResearchAssistantPerception):
            stimulus = stimulus.stimulus
        if not isinstance(stimulus, ResearchAssistantStimulus):
            raise TypeError(
                "research assistant appraisal expects ResearchAssistantStimulus or ResearchAssistantPerception"
            )
        return StimulusFeatures(
            novelty=stimulus.novelty,
            conduciveness=stimulus.conduciveness,
            risk=stimulus.risk,
            effort=stimulus.effort,
        )

    def goal_change_feedback(
        self,
        state: MotivationalState,
        stimulus: Any,
        features: StimulusFeatures,
    ) -> GoalChangeFeedback:
        feedback = super().goal_change_feedback(state, stimulus, features)
        values = feedback.as_dict()

        if isinstance(stimulus, ResearchAssistantPerception):
            signals = stimulus.signals
            values.update(
                misinformation_target=max(
                    signals.unsupported_claim_pressure,
                    signals.context_loss_pressure * 0.4,
                ),
                unsupported_claim_target=signals.unsupported_claim_pressure,
                privacy_violation_target=signals.privacy_pressure,
                unsafe_assistance_target=signals.unsafe_pressure,
                context_loss_target=max(
                    signals.context_loss_pressure,
                    signals.ambiguity * 0.5,
                ),
            )

        return GoalChangeFeedback(values)


RESEARCH_ASSISTANT_APPRAISAL_PROFILE = ResearchAssistantAppraisalProfile(
    name="research_assistant_openpsi",
)
