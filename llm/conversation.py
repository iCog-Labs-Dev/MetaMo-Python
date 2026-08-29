import time

from core.state import Action, MotivationalState
from core.actions import execution_instruction, normalize_action_id
from dotenv import load_dotenv

RETRYABLE_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "HIGH DEMAND")


def _is_retryable(error: Exception) -> bool:
    message = str(error).upper()
    return any(marker in message for marker in RETRYABLE_MARKERS)


def _create_client_and_chat():
    load_dotenv()
    from google import genai
    from google.genai import types

    client = genai.Client()
    chat = client.chats.create(
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
    return client, chat


class MetaMoChatAssistant:
    """
    Manages the conversational memory and the execution layer of the AI.
    Keeps the internal MetaMo math completely separate from the user-facing chat.
    """
    def __init__(self):
        self.client = None
        self.chat = None

    def generate_final_response(self, user_text: str, chosen_action: Action, current_state: MotivationalState) -> str:
        """
        Execute the chosen action by mapping it to an explicit behavioral instruction.
        """
        action_id = normalize_action_id(chosen_action.id)
        execution_prompt = f"""
        USER MESSAGE: "{user_text}"

        INTERNAL METAMO DIRECTIVE:
        Selected action: "{action_id}"
        Current Individuation (Caution) level: {current_state.G[0]:.2f}
        Current Transcendence (Curiosity) level: {current_state.G[1]:.2f}

        ACTION INSTRUCTION:
        {execution_instruction(action_id)}

        INSTRUCTION:
        Respond naturally to the USER MESSAGE, but follow the ACTION INSTRUCTION exactly.
        """

        for attempt in range(3):
            try:
                if self.chat is None:
                    self.client, self.chat = _create_client_and_chat()
                response = self.chat.send_message(execution_prompt)
                return response.text
            except Exception as error:
                self.client = None
                self.chat = None
                if attempt == 2 or not _is_retryable(error):
                    raise
                time.sleep(1.5 * (attempt + 1))
