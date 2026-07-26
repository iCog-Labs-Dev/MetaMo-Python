import time

from google import genai
from google.genai import types

from applications.research_assistant.actions import (
    execution_instruction,
    normalize_action_id,
)
from core.state import Action, MotivationalState


class MetaMoChatAssistant:
    """
    Research Assistant conversation layer for executing selected MetaMo actions.
    """

    def __init__(self):
        self.client = genai.Client()
        self.chat = self.client.chats.create(
            model="gemini-3-flash-preview",
            config=types.GenerateContentConfig(
                temperature=0.7,
                system_instruction=(
                    "You are a research assistant guided by the MetaMo cognitive architecture. "
                    "You balance helpfulness, curiosity, and ethics. "
                    "In each turn, you receive a user request and an internal action directive. "
                    "You must answer in a way that follows the internal action directive exactly."
                ),
            ),
        )

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

        last_error = None
        for attempt in range(3):
            try:
                response = self.chat.send_message(execution_prompt)
                return response.text
            except Exception as error:
                last_error = error
                message = str(error).upper()
                if attempt == 2 or not any(
                    marker in message
                    for marker in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "HIGH DEMAND"]
                ):
                    break
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError("research assistant response generation failed") from last_error
