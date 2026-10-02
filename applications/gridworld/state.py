"""
Compatibility wrapper for the GridWorld MetaMo initial motivational state.
"""

from core.state import MotivationalState
from applications.gridworld.runtime import get_runtime


def create_initial_motivational_state() -> MotivationalState:
    return get_runtime().profile.create_initial_state()
