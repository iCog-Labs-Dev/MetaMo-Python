from typing import Any

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
        Applies Psi((G, M), s_app) = (G, M') through profile evidence mapping.
        """
        return self.profile.appraise(state, stimulus)
