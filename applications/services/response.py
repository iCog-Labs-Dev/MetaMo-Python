from dataclasses import dataclass


@dataclass
class AssistantResponse:
    text: str
    action_id: str
    individuation: float
    transcendence: float
    curiosity_action: str
    ethics_action: str
    simulated_caution: float


def format_response(response: AssistantResponse) -> str:
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
