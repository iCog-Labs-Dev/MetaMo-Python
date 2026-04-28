NUM_GOALS = 8
G_IND, G_TRANS = 0, 1
G_HELP, G_CURIO, G_NOVEL, G_SELF, G_ETHIC, G_SOC = 2, 3, 4, 5, 6, 7

NUM_MODULATORS = 6
M_VALENCE, M_AROUSAL, M_APPROACH = 0, 1, 2
M_RESOLUTION, M_THRESHOLD, M_SECURING = 3, 4, 5

LAMBDA_IND = 0.5  # Weight of the individuation penalty (suppresses risk).
LAMBDA_TRANS = 0.5  # Weight of the transcendence reward (encourages growth).

THETA_SAFE = 0.3  # Minimum required level of individuation for safety.
G_MAX = 2.0

# Contractive update law parameters for states near the boundary[cite: 132, 176].
# d(F(x), F(y)) <= C_CONTRACT * d(x, y) + EPSILON
C_CONTRACT = 0.9  # Must be < 1 to ensure contractivity[cite: 132].
EPSILON = 0.05  # Small allowed error margin[cite: 132].
ETA_BOUNDARY = (
    0.1  # Distance from the edge that triggers the boundary band (B_eta)[cite: 383].
)

ALPHA_0 = 0.1  # Base rate slowed down by individuation[cite: 135].
BETA_0 = 0.15  # Base rate sped up by transcendence[cite: 135].

LAX_DISTRIBUTIVE_DELTA = 1e-3
