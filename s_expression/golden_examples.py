"""Golden examples for the MetaMo S-expression serializer."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.config import NUM_GOALS, NUM_MODULATORS
from core.serializer import to_sexpr
from core.state import Action, MotivationalState, Stimulus


def build_meta_mo_examples() -> dict[str, Any]:
    state = MotivationalState(
        G=np.array([0.5, 0.5, 0.8, 0.6, 0.4, 0.3, 0.9, 0.2], dtype=float),
        M=np.full(NUM_MODULATORS, 0.5, dtype=float),
    )
    stimulus = Stimulus(novelty=0.8, conduciveness=0.5, risk=0.2, effort=0.3)
    action = Action(
        id="explore",
        goal_correlations=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=float),
        risk_estimate=0.6,
        delta_g=np.array([0.05, -0.05, 0.02, -0.02, 0.01, -0.01, 0.03, -0.03], dtype=float),
    )
    candidate_actions = [action]
    return {
        "state": state,
        "stimulus": stimulus,
        "action": action,
        "candidates": candidate_actions,
    }


def golden_sexpr_examples() -> dict[str, str]:
    examples = build_meta_mo_examples()
    return {
        "state": to_sexpr(examples["state"]),
        "stimulus": to_sexpr(examples["stimulus"]),
        "action": to_sexpr(examples["action"]),
        "candidates": to_sexpr(examples["candidates"]),
    }


if __name__ == "__main__":
    for name, text in golden_sexpr_examples().items():
        print(f";; {name}\n{text}\n")
