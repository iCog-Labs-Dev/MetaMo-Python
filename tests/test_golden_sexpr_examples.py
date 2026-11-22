"""Golden examples for the generic S-expression serializer."""

import numpy as np

from core.config import NUM_GOALS, NUM_MODULATORS
from core.serializer import parse_sexpr, to_sexpr
from core.state import Action, MotivationalState


def test_meta_mo_golden_examples_round_trip() -> None:
    state = MotivationalState(
        G=np.full(NUM_GOALS, 0.5, dtype=float),
        M=np.full(NUM_MODULATORS, 0.5, dtype=float),
    )
    action = Action(
        id="explore",
        goal_correlations=np.full(NUM_GOALS, 0.1, dtype=float),
        delta_g=np.full(NUM_GOALS, 0.05, dtype=float),
        metadata={"generated_by": "metamo-test"},
    )

    state_text = to_sexpr(state)
    action_text = to_sexpr(action)
    action_list_text = to_sexpr([action])

    assert "MotivationalState" in state_text
    assert "Action" in action_text
    assert "explore" in action_text
    assert parse_sexpr(state_text)
    assert parse_sexpr(action_text)
    assert parse_sexpr(action_list_text)

    assert to_sexpr(state) == state_text
    assert to_sexpr(action) == action_text
    assert to_sexpr([action]) == action_list_text
