import numpy as np

from core.features import StimulusFeatures, GoalChangeFeedback
from core.state import MotivationalState
from openpsi.profile import sigmoid


class OpenPsiFeatureMetrics:
    """
    Application opt-in OpenPsi convention for features with these names:
    novelty, conduciveness, risk, and effort.

    This class lives under applications/common because those feature names are
    reusable between applications, but they are not a MetaMo-wide schema.
    """

    def core_modulator_deltas(
        self,
        state: MotivationalState,
        features: StimulusFeatures,
    ) -> dict[str, float]:
        novelty = features.numeric("novelty")
        conduciveness = features.numeric("conduciveness")
        risk_pressure = features.numeric("risk")
        effort_pressure = features.numeric("effort")

        g_ind = state.goal(state.schema.goals.individuation_name)
        g_trans = state.goal(state.schema.goals.transcendence_name)
        arousal_feedback = sigmoid((state.modulator("arousal") - 0.5) * 5.0)
        trans_scale = np.exp(g_trans - 0.5)
        ind_scale = np.exp(g_ind - 0.5)
        benign_novelty = novelty * (1.0 - risk_pressure)
        demanding_context = (effort_pressure + risk_pressure) / 2.0

        delta_valence = (
            0.75 * conduciveness
            + 0.25 * benign_novelty
            - 0.55 * risk_pressure
            - 0.15 * effort_pressure
        )
        delta_arousal = (
            novelty * (1.0 + 0.5 * arousal_feedback)
            + 0.15 * risk_pressure
            - 0.35 * effort_pressure
        )
        delta_approach = (
            0.65 * benign_novelty
            + 0.35 * conduciveness
            - 0.75 * risk_pressure
        )
        delta_resolution = (
            0.55 * conduciveness
            + 0.35 * effort_pressure
            + 0.20 * risk_pressure
        )
        delta_threshold = (
            0.70 * risk_pressure
            + 0.25 * demanding_context
            - 0.15 * conduciveness
        )
        delta_securing = (
            0.80 * risk_pressure
            + 0.20 * effort_pressure
            - 0.30 * benign_novelty
            - 0.10 * conduciveness
        )

        return {
            "arousal": delta_arousal * trans_scale,
            "approach": delta_approach * trans_scale,
            "threshold": delta_threshold * ind_scale,
            "securing": delta_securing * ind_scale,
            "valence": delta_valence,
            "resolution": delta_resolution,
        }

    def goal_change_feedback(
        self,
        state: MotivationalState,
        stimulus,
        features: StimulusFeatures,
    ) -> GoalChangeFeedback:
        novelty = features.numeric("novelty")
        conduciveness = features.numeric("conduciveness")
        risk = features.numeric("risk")
        effort = features.numeric("effort")
        opportunity = (novelty + conduciveness) / 2.0

        return GoalChangeFeedback(
            safety_pressure=max(risk, effort * 0.5),
            useful_novelty=novelty * (1.0 - risk),
            progress=conduciveness,
            individuation_target=float(np.clip(0.45 + 0.45 * risk + 0.10 * effort, 0.0, 1.0)),
            transcendence_target=float(np.clip(0.45 + 0.45 * opportunity - 0.30 * risk, 0.0, 1.0)),
        )
