from core.schema import DEFAULT_MOTIVATION_SCHEMA


NUM_GOALS = DEFAULT_MOTIVATION_SCHEMA.num_goals
G_IND = DEFAULT_MOTIVATION_SCHEMA.goal_index("individuation")
G_TRANS = DEFAULT_MOTIVATION_SCHEMA.goal_index("transcendence")
G_HELP = DEFAULT_MOTIVATION_SCHEMA.goal_index("help")
G_CURIO = DEFAULT_MOTIVATION_SCHEMA.goal_index("curiosity")
G_NOVEL = DEFAULT_MOTIVATION_SCHEMA.goal_index("novelty")
G_SELF = DEFAULT_MOTIVATION_SCHEMA.goal_index("self_improvement")
G_ETHIC = DEFAULT_MOTIVATION_SCHEMA.goal_index("ethics")
G_SOC = DEFAULT_MOTIVATION_SCHEMA.goal_index("sociality")

NUM_MODULATORS = DEFAULT_MOTIVATION_SCHEMA.num_modulators
M_VALENCE = DEFAULT_MOTIVATION_SCHEMA.modulator_index("valence")
M_AROUSAL = DEFAULT_MOTIVATION_SCHEMA.modulator_index("arousal")
M_APPROACH = DEFAULT_MOTIVATION_SCHEMA.modulator_index("approach")
M_RESOLUTION = DEFAULT_MOTIVATION_SCHEMA.modulator_index("resolution")
M_THRESHOLD = DEFAULT_MOTIVATION_SCHEMA.modulator_index("threshold")
M_SECURING = DEFAULT_MOTIVATION_SCHEMA.modulator_index("securing")

LAMBDA_IND = 0.5  # Weight of the individuation penalty (suppresses risk).
LAMBDA_TRANS = 0.5  # Weight of the transcendence reward (encourages growth).

THETA_SAFE = 0.3  # Minimum required level of individuation for safety.
G_MAX = 2.0

# Contractive update law parameters for states near the boundary.
# d(F(x), F(y)) <= C_CONTRACT * d(x, y) + EPSILON
C_CONTRACT = 0.9  # Must be < 1 to ensure contractivity.
EPSILON = 0.05  # Small allowed error margin.
ETA_BOUNDARY = (
    0.1  # Distance from the edge that triggers the boundary band (B_eta).

)

ALPHA_0 = 0.1  # Base rate slowed down by individuation.
BETA_0 = 0.15  # Base rate sped up by transcendence.

LAX_DISTRIBUTIVE_DELTA = 1e-2
PARALLEL_COMPOSITION_DELTA = 1e-2
