from typing import Any

from core.features import GoalChangeFeedback
from core.state import MotivationalState
from category.functors import AppraisalComonad
from openpsi.profile import AppraisalProfile


class OpenPsiAppraisal(AppraisalComonad):
    """
    Configurable OpenPsi appraisal layer as the comonad Psi.
    """

    def __init__(self, profile: AppraisalProfile):
        self.profile = profile

    def extract(self, state: MotivationalState) -> MotivationalState:
        """
        The comonadic counit. Extracts the current state.
        """
        return state

    def appraise(self, state: MotivationalState, stimulus: Any) -> MotivationalState:
        """
        Applies Psi((G, M), s) = (G, M') through profile feature mapping.
        """
        return self.profile.appraise(state, stimulus)

    def goal_change_feedback(
        self,
        state: MotivationalState,
        stimulus: Any,
    ) -> GoalChangeFeedback:
        features = self.profile.stimulus_features(state, stimulus)
        return self.profile.goal_change_feedback(state, stimulus, features)

    def appraise_with_feedback(
        self,
        state: MotivationalState,
        stimulus: Any,
    ) -> tuple[MotivationalState, GoalChangeFeedback]:
        """Extract application features once for appraisal and goal feedback."""
        features = self.profile.stimulus_features(state, stimulus)
        appraised_state = self.profile.appraise_features(state, features)
        feedback = self.profile.goal_change_feedback(state, stimulus, features)
        return appraised_state, feedback
