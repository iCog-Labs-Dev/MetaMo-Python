from dataclasses import dataclass

from applications.research_assistant.decision_profile import (
    RESEARCH_ASSISTANT_DECISION_PROFILE,
)
from applications.research_assistant.appraisal_profile import (
    RESEARCH_ASSISTANT_APPRAISAL_PROFILE,
)
from applications.research_assistant.schema import RESEARCH_ASSISTANT_SCHEMA
from applications.research_assistant.state import make_state
from applications.research_assistant.translations import (
    curiosity_to_ethics_goal_translation,
    curiosity_to_ethics_modulator_translation,
)
from category.bimonad import MetaMoPseudoBimonad
from category.diagnostics import MetaMoDiagnosticsSummary
from category.functors import TranslationFunctor
from core.state import MotivationalState
from applications.research_assistant.calibration import calibrate_perception
from applications.research_assistant.llm.client import get_perception_from_text
from applications.research_assistant.llm.conversation import MetaMoChatAssistant
from applications.research_assistant.planning import build_candidates
from magus.decision import MagusDecision
from openpsi.appraisal import OpenPsiAppraisal


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
    simulation_error: float
    lax_error: float
    self_model_drift: float
    calibration_adjustments: tuple[str, ...] = ()


def format_response(response: AssistantResponse) -> str:
    """Render an AssistantResponse as a human-readable string."""
    return (
        f"  > [Curiosity Subsystem] wants to: {response.curiosity_action}\n"
        f"  > [Ethics Subsystem] wants to: {response.ethics_action}\n"
        f"  > [Reciprocal Simulation]: Curiosity predicts Ethics caution "
        f"{response.simulated_caution:.2f} (error {response.simulation_error:.3f})\n"
        f"  > [Continuity]: lax error {response.lax_error:.3f}, "
        f"self-model drift {response.self_model_drift:.3f}\n"
        f"\n"
        f"Assistant: {response.text}\n"
        f"\n"
        f"[Consensus State -> Individuation: {response.individuation:.2f} "
        f"| Transcendence: {response.transcendence:.2f}]"
    )


class MetaMoEngine:
    """Orchestrates the Research Assistant MetaMo pipeline for one user input."""

    def __init__(self):
        decision = MagusDecision(profile=RESEARCH_ASSISTANT_DECISION_PROFILE)
        appraisal = OpenPsiAppraisal(profile=RESEARCH_ASSISTANT_APPRAISAL_PROFILE)
        self.bimonad = MetaMoPseudoBimonad(appraisal, decision)
        self.assistant = MetaMoChatAssistant()
        self.translator = TranslationFunctor(
            goal_translation=curiosity_to_ethics_goal_translation(),
            modulator_translation=curiosity_to_ethics_modulator_translation(),
            target_schema=RESEARCH_ASSISTANT_SCHEMA,
        )
        self.state_curiosity = self._make_state(override={"transcendence": 0.9})
        self.state_ethics = self._make_state(override={"individuation": 0.9})

    @staticmethod
    def _make_state(override: dict[str | int, float] | None = None) -> MotivationalState:
        return make_state(override=override)

    def process(self, user_input: str) -> AssistantResponse:
        """Run the full MetaMo pipeline on *user_input* and return the result."""
        calibration = calibrate_perception(get_perception_from_text(user_input))
        perception = calibration.perception
        merged_current = self.bimonad.parallel_merge(self.state_curiosity, self.state_ethics)
        current_mood = {
            "arousal": merged_current.modulator("arousal"),
            "caution": merged_current.modulator("securing"),
        }
        candidates = build_candidates(perception, current_mood)

        action_c, next_curiosity, diagnostics_c = self.bimonad.step_with_diagnostics(
            self.state_curiosity,
            perception,
            candidates,
        )

        action_e, next_ethics, diagnostics_e = self.bimonad.step_with_diagnostics(
            self.state_ethics,
            perception,
            candidates,
        )

        simulation = self.translator.check_reciprocal_simulation(
            lambda state, stim, acts: self.bimonad.step(
                state,
                stim,
                acts,
                record_diagnostics=False,
            ),
            lambda state, stim, acts: self.bimonad.step(
                state,
                stim,
                acts,
                record_diagnostics=False,
            ),
            self.state_curiosity,
            perception,
            candidates,
        )
        simulated_ethics = simulation.target_after_translation_update

        final_action, merged_target = self.bimonad.consensus_transition(
            self.state_curiosity,
            self.state_ethics,
            perception,
            candidates,
        )

        response_text = self.assistant.generate_final_response(user_input, final_action, merged_target)

        self.state_curiosity = next_curiosity
        self.state_ethics = next_ethics

        return AssistantResponse(
            text=response_text,
            action_id=final_action.id,
            individuation=merged_target.goal("individuation"),
            transcendence=merged_target.goal("transcendence"),
            curiosity_action=action_c.id,
            ethics_action=action_e.id,
            simulated_caution=float(
                (
                    simulated_ethics.modulator("threshold")
                    + simulated_ethics.modulator("securing")
                )
                / 2.0
            ),
            simulation_error=float(simulation.error),
            lax_error=float(max(diagnostics_c.lax_error, diagnostics_e.lax_error)),
            self_model_drift=float(max(diagnostics_c.self_model_drift, diagnostics_e.self_model_drift)),
            calibration_adjustments=calibration.adjustments,
        )

    def process_with_context(self, user_input: str, context: str) -> AssistantResponse:
        """Prepend *context* to *user_input* before processing."""
        augmented_input = f"[Paper Context]\n{context}\n\n[User Query]\n{user_input}"
        return self.process(augmented_input)

    def diagnostics_summary(self) -> MetaMoDiagnosticsSummary:
        """Return aggregate MetaMo diagnostics for this engine session."""
        return self.bimonad.diagnostics_history.summary()

    def write_diagnostics_csv(self, path: str) -> None:
        """Persist session diagnostics to a CSV file."""
        self.bimonad.diagnostics_history.write_csv(path)

    def reset_diagnostics(self) -> None:
        """Clear accumulated session diagnostics."""
        self.bimonad.diagnostics_history.clear()
