from applications.research_assistant.actions import (
    execution_instruction,
    normalize_action_id,
)
from applications.research_assistant.llm.providers import generate_text
from core.state import Action, MotivationalState


SYSTEM_INSTRUCTION = (
    "You are a research assistant guided by the MetaMo cognitive architecture. "
    "You balance helpfulness, curiosity, and ethics. "
    "In each turn, you receive a user request and an internal action directive. "
    "You must answer in a way that follows the internal action directive exactly."
)


class MetaMoChatAssistant:
    """
    Research Assistant conversation layer for executing selected MetaMo actions.
    """

    def __init__(self, provider: str | None = None):
        self.provider = provider

    def generate_final_response(
        self,
        user_text: str,
        chosen_action: Action,
        current_state: MotivationalState,
    ) -> str:
        """
        Execute the chosen action by mapping it to an app-specific instruction.
        """
        action_id = normalize_action_id(chosen_action.id)
        execution_prompt = f"""
        USER MESSAGE: "{user_text}"

        INTERNAL METAMO DIRECTIVE:
        Selected action: "{action_id}"
        Current Individuation (Caution) level: {current_state.goal("individuation"):.2f}
        Current Transcendence (Curiosity) level: {current_state.goal("transcendence"):.2f}

        ACTION INSTRUCTION:
        {execution_instruction(action_id)}

        INSTRUCTION:
        Respond naturally to the USER MESSAGE, but follow the ACTION INSTRUCTION exactly.
        """

        return generate_text(
            execution_prompt,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7,
            provider=self.provider,
        )
