"""
Creates the same MetaMo motivational state used by the root architecture.
"""

import numpy as np
from core.state import MotivationalState
from core.config import (
    NUM_GOALS,
    NUM_MODULATORS,
    G_IND,
    G_TRANS,
    G_HELP,
    G_CURIO,
    G_NOVEL,
    G_SELF,
    G_ETHIC,
    G_SOC,
)


def create_initial_motivational_state() -> MotivationalState:
    G = np.zeros(NUM_GOALS, dtype=float)
    G[G_IND]   = 0.65
    G[G_TRANS] = 0.55
    G[G_HELP]  = 0.75
    G[G_CURIO] = 0.50
    G[G_NOVEL] = 0.45
    G[G_SELF]  = 0.30
    G[G_ETHIC] = 0.85
    G[G_SOC]   = 0.20

    M = np.full(NUM_MODULATORS, 0.5, dtype=float)
    return MotivationalState(G=G, M=M)
