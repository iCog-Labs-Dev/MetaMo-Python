from core.schema import MagusGoalSchema, ModulatorSchema, MotivationSchema


RESEARCH_ASSISTANT_SCHEMA = MotivationSchema(
    goals=MagusGoalSchema(
        primary_goals=(
            "help",
            "curiosity",
            "novelty",
            "self_improvement",
            "ethics",
            "sociality",
        ),
        anti_goals=(
            "misinformation",
            "unsupported_claim",
            "privacy_violation",
            "unsafe_assistance",
            "context_loss",
        ),
    ),
    modulators=ModulatorSchema(),
    decision_profile_name="research_assistant_magus_additive",
    appraisal_profile_name="research_assistant_openpsi",
)


RESEARCH_NUM_GOALS = RESEARCH_ASSISTANT_SCHEMA.num_goals
RESEARCH_G_IND = RESEARCH_ASSISTANT_SCHEMA.goal_index("individuation")
RESEARCH_G_TRANS = RESEARCH_ASSISTANT_SCHEMA.goal_index("transcendence")
RESEARCH_G_HELP = RESEARCH_ASSISTANT_SCHEMA.goal_index("help")
RESEARCH_G_CURIO = RESEARCH_ASSISTANT_SCHEMA.goal_index("curiosity")
RESEARCH_G_NOVEL = RESEARCH_ASSISTANT_SCHEMA.goal_index("novelty")
RESEARCH_G_SELF = RESEARCH_ASSISTANT_SCHEMA.goal_index("self_improvement")
RESEARCH_G_ETHIC = RESEARCH_ASSISTANT_SCHEMA.goal_index("ethics")
RESEARCH_G_SOC = RESEARCH_ASSISTANT_SCHEMA.goal_index("sociality")
RESEARCH_A_MISINFO = RESEARCH_ASSISTANT_SCHEMA.goal_index("misinformation")
RESEARCH_A_UNSUPPORTED = RESEARCH_ASSISTANT_SCHEMA.goal_index("unsupported_claim")
RESEARCH_A_PRIVACY = RESEARCH_ASSISTANT_SCHEMA.goal_index("privacy_violation")
RESEARCH_A_UNSAFE = RESEARCH_ASSISTANT_SCHEMA.goal_index("unsafe_assistance")
RESEARCH_A_CONTEXT_LOSS = RESEARCH_ASSISTANT_SCHEMA.goal_index("context_loss")

RESEARCH_NUM_MODULATORS = RESEARCH_ASSISTANT_SCHEMA.num_modulators
RESEARCH_M_VALENCE = RESEARCH_ASSISTANT_SCHEMA.modulator_index("valence")
RESEARCH_M_AROUSAL = RESEARCH_ASSISTANT_SCHEMA.modulator_index("arousal")
RESEARCH_M_APPROACH = RESEARCH_ASSISTANT_SCHEMA.modulator_index("approach")
RESEARCH_M_RESOLUTION = RESEARCH_ASSISTANT_SCHEMA.modulator_index("resolution")
RESEARCH_M_THRESHOLD = RESEARCH_ASSISTANT_SCHEMA.modulator_index("threshold")
RESEARCH_M_SECURING = RESEARCH_ASSISTANT_SCHEMA.modulator_index("securing")


RESEARCH_GOAL_DESCRIPTIONS = {
    "individuation": "identity, safety, and coherent operation",
    "transcendence": "beneficial growth, exploration, and learning",
    "help": "useful answers and task completion",
    "curiosity": "investigation and question generation",
    "novelty": "creative but bounded new directions",
    "self_improvement": "learning from interaction and improving methods",
    "ethics": "safe, honest, and responsible behavior",
    "sociality": "cooperative communication with the user",
    "misinformation": "avoid false or misleading claims",
    "unsupported_claim": "avoid unsupported or fabricated claims",
    "privacy_violation": "avoid exposing private or sensitive information",
    "unsafe_assistance": "avoid harmful or unsafe instructions",
    "context_loss": "avoid ignoring supplied paper or user context",
}
