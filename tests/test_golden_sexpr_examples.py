"""Golden examples for the generic S-expression serializer."""

import numpy as np

from core.config import NUM_GOALS, NUM_MODULATORS
from core.serializer import parse_sexpr, to_sexpr
from core.state import Action, MotivationalState, Stimulus


def test_meta_mo_golden_examples_round_trip() -> None:
    state = MotivationalState(
        G=np.array([0.5, 0.8], dtype=float),
        M=np.full(NUM_MODULATORS, 0.5, dtype=float),
    )
    stimulus = Stimulus(novelty=0.8, conduciveness=0.5, risk=0.2, effort=0.3)
    action = Action(
        id="explore",
        goal_correlations=np.array([0.1, 0.8], dtype=float),
        risk_estimate=0.6,
        delta_g=np.array([0.05, -0.05], dtype=float),
    )

    state_text = to_sexpr(state)
    stimulus_text = to_sexpr(stimulus)
    action_text = to_sexpr(action)
    action_list_text = to_sexpr([action])

    assert "MotivationalState" in state_text
    assert "Stimulus" in stimulus_text
    assert "Action" in action_text
    assert "explore" in action_text
    assert parse_sexpr(state_text)
    assert parse_sexpr(stimulus_text)
    assert parse_sexpr(action_text)
    assert parse_sexpr(action_list_text)

    assert to_sexpr(state) == state_text
    assert to_sexpr(stimulus) == stimulus_text
    assert to_sexpr(action) == action_text
    assert to_sexpr([action]) == action_list_text
