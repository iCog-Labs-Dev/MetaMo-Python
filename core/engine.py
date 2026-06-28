from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.state import MotivationalState
from core.config import (
    G_IND, G_TRANS, G_HELP, G_CURIO, G_NOVEL, G_SELF, G_ETHIC, G_SOC,
    M_AROUSAL, M_SECURING, NUM_GOALS, NUM_MODULATORS,
)
from category.functors import TranslationFunctor
from category.bimonad import MetaMoPseudoBimonad
from dynamics.coherence import blend_states
from openpsi.appraisal import OpenPsiAppraisal
from magus.decision import MagusDecision
from llm.client import get_stimulus_from_text, get_candidates_from_text
from llm.conversation import MetaMoChatAssistant


@dataclass
class AssistantResponse:
    """Encapsulates the full output of a MetaMo processing cycle for display."""

    text: str
    action_id: str
    individuation: float
    transcendence: float
    curiosity_action: str
    ethics_action: str
    simulated_caution: float


def format_response(response: AssistantResponse) -> str:
    """Render an AssistantResponse as a human-readable string."""
    return (
        f"  > [Curiosity Subsystem] wants to: {response.curiosity_action}\n"
        f"  > [Ethics Subsystem] wants to: {response.ethics_action}\n"
        f"  > [Reciprocal Simulation]: Curiosity agent predicts Ethics agent's caution is {response.simulated_caution:.2f}\n"
        f"\n"
        f"Assistant: {response.text}\n"
        f"\n"
        f"[Consensus State -> Individuation: {response.individuation:.2f} "
        f"| Transcendence: {response.transcendence:.2f}]"
    )


def _default_goal_vector() -> np.ndarray:
    """Return the default goal vector used when initialising states."""
    G = np.zeros(NUM_GOALS)
    G[G_IND] = 0.5
    G[G_TRANS] = 0.5
    G[G_HELP] = 0.8
    G[G_CURIO] = 0.6
    G[G_ETHIC] = 0.9
    G[G_NOVEL] = 0.4
    G[G_SELF] = 0.3
    G[G_SOC] = 0.2
    return G


class MetaMoEngine:
    """Orchestrates the full MetaMo processing pipeline for a single user input."""

    def __init__(self):
        self.bimonad = MetaMoPseudoBimonad(OpenPsiAppraisal(), MagusDecision())
        self.assistant = MetaMoChatAssistant()
        self.translator = TranslationFunctor(
            goal_translation=np.eye(NUM_GOALS),
            modulator_translation=np.eye(NUM_MODULATORS),
        )
        self.state_curiosity = self._make_state(override={G_TRANS: 0.9})
        self.state_ethics = self._make_state(override={G_IND: 0.9})

    @staticmethod
    def _make_state(override: Optional[dict] = None) -> MotivationalState:
        """Build a MotivationalState with optional goal-index overrides."""
        G = _default_goal_vector()
        if override:
            for idx, val in override.items():
                G[idx] = val
        M = np.full(NUM_MODULATORS, 0.5)
        return MotivationalState(G=G, M=M)

    def process(self, user_input: str) -> AssistantResponse:
        """Run the full MetaMo pipeline on *user_input* and return the result."""
        stimulus = get_stimulus_from_text(user_input)
        merged_current = self.bimonad.parallel_merge(self.state_curiosity, self.state_ethics)
        current_mood = {"arousal": merged_current.M[M_AROUSAL], "caution": merged_current.M[M_SECURING]}
        candidates = get_candidates_from_text(user_input, current_mood)

        action_c, target_c = self.bimonad.step(self.state_curiosity, stimulus, candidates)
        action_e, target_e = self.bimonad.step(self.state_ethics, stimulus, candidates)

        simulated_ethics = self.translator.simulate_peer(self.state_curiosity)

        final_action, merged_target = self.bimonad.consensus_transition(
            self.state_curiosity, self.state_ethics, stimulus, candidates,
        )

        response_text = self.assistant.generate_final_response(user_input, final_action, merged_target)

        self.state_curiosity = blend_states(self.state_curiosity, target_c)
        self.state_ethics = blend_states(self.state_ethics, target_e)

        return AssistantResponse(
            text=response_text,
            action_id=final_action.id,
            individuation=merged_target.G[G_IND],
            transcendence=merged_target.G[G_TRANS],
            curiosity_action=action_c.id,
            ethics_action=action_e.id,
            simulated_caution=float(simulated_ethics.G[G_IND]),
        )

    def process_with_context(self, user_input: str, context: str) -> AssistantResponse:
        """Prepend *context* to *user_input* before processing."""
        augmented_input = f"[Paper Context]\n{context}\n\n[User Query]\n{user_input}"
        return self.process(augmented_input)
