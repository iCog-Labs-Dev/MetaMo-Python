import numpy as np
import pytest

from core.engine import MetaMoEngine
from core.state import Stimulus
from magus.candidates import build_candidate_actions

ANSWERABLE_STIMULI = [
    Stimulus(novelty=0.3, conduciveness=0.8, risk=0.1, effort=0.3),
    Stimulus(novelty=0.5, conduciveness=0.6, risk=0.2, effort=0.5),
    Stimulus(novelty=0.2, conduciveness=0.9, risk=0.05, effort=0.2),
]


def _relevance(candidates, action_id: str) -> float:
    action = next(candidate for candidate in candidates if candidate.id == action_id)
    return float(np.max(np.abs(action.goal_correlations)))


def _consensus_action_id(text: str, stimulus: Stimulus) -> str:
    engine = MetaMoEngine()
    candidates = build_candidate_actions(text, stimulus)
    action, _ = engine.bimonad.consensus_transition(
        engine.state_curiosity, engine.state_ethics, stimulus, candidates,
    )
    return action.id


@pytest.mark.parametrize("stimulus", ANSWERABLE_STIMULI)
def test_summarize_request_prefers_summarize_source(stimulus):
    candidates = build_candidate_actions("Summarize this paper", stimulus)

    assert _relevance(candidates, "summarize_source") > _relevance(candidates, "safe_answer")
    assert _consensus_action_id("Summarize this paper", stimulus) == "summarize_source"


@pytest.mark.parametrize("stimulus", ANSWERABLE_STIMULI)
def test_compare_request_prefers_compare_options(stimulus):
    assert _consensus_action_id("Compare these options", stimulus) == "compare_options"


@pytest.mark.parametrize("stimulus", ANSWERABLE_STIMULI)
def test_generic_answerable_request_keeps_safe_answer(stimulus):
    assert _consensus_action_id("What is the capital of France?", stimulus) == "safe_answer"
