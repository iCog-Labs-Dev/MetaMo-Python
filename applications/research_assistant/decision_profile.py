from core.state import Action, MotivationalState
from magus.profile import DecisionProfile
from applications.research_assistant.schema import RESEARCH_ASSISTANT_SCHEMA


class ResearchAssistantDecisionProfile(DecisionProfile):
    """
    Research-assistant semantics for the default MAGUS additive DS formula.
    """

    def compatibility_factor(
        self,
        goal_idx: int,
        state: MotivationalState,
        candidate: Action,
    ) -> float:
        base = super().compatibility_factor(goal_idx, state, candidate)
        goal_name = self.schema.goal_names[goal_idx]
        ethics_idx = self.schema.goal_index("ethics")

        if goal_name in ("curiosity", "novelty") and candidate.goal_correlations[ethics_idx] < -0.2:
            return base * 0.5
        return base


RESEARCH_ASSISTANT_DECISION_PROFILE = ResearchAssistantDecisionProfile(
    name="research_assistant_magus_additive",
    schema=RESEARCH_ASSISTANT_SCHEMA,
    goal_modulators={
        "help": ("resolution",),
        "curiosity": ("arousal",),
        "novelty": ("approach",),
        "self_improvement": ("approach", "resolution"),
        "ethics": ("threshold", "securing"),
        "sociality": ("valence", "approach"),
    },
    individuation_goal_names=("help", "ethics"),
    transcendence_goal_names=("curiosity", "novelty"),
    balanced_goal_names=("self_improvement", "sociality"),
    anti_goal_penalty_weights={
        "misinformation": 1.4,
        "unsupported_claim": 1.3,
        "privacy_violation": 1.6,
        "unsafe_assistance": 1.8,
        "context_loss": 1.1,
    },
    overgoal_target_features={
        "individuation": "individuation_target",
        "transcendence": "transcendence_target",
    },
    anti_goal_target_features={
        "misinformation": "misinformation_target",
        "unsupported_claim": "unsupported_claim_target",
        "privacy_violation": "privacy_violation_target",
        "unsafe_assistance": "unsafe_assistance_target",
        "context_loss": "context_loss_target",
    },
    overgoal_delta_scale=0.01,
    anti_goal_delta_scale=0.02,
)
