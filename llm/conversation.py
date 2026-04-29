import time

from google import genai
from google.genai import types
from core.state import Action, MotivationalState
from llm.action_schema import execution_instruction, normalize_action_id

class MetaMoChatAssistant:
    """
    Manages the conversational memory and the execution layer of the AI.
    Keeps the internal MetaMo math completely separate from the user-facing chat.
    """
    def __init__(self):
        # Initialize the Gemini client
        self.client = genai.Client()
        self.chat = self.client.chats.create(
            model='gemini-3-flash-preview',
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

        last_error = None
        for attempt in range(3):
            try:
                response = self.chat.send_message(execution_prompt)
                return response.text
            except Exception as error:
                last_error = error
                message = str(error).upper()
                if attempt == 2 or not any(marker in message for marker in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "HIGH DEMAND"]):
                    break
                time.sleep(1.5 * (attempt + 1))

        if action_id == "ask_clarifying_question":
            return "I need one short clarification before I answer: what part do you want me to focus on?"
        if action_id == "compare_options":
            return "I cannot reach the external response model right now, but I would compare the main options and explain their tradeoffs."
        if action_id == "summarize_source":
            return "I cannot reach the external response model right now, but I would give a careful summary of the source."
        if action_id == "decline_risky_request":
            return "I cannot help with a risky request, but I can help with a safer alternative."
        if action_id == "guided_explore":
            return "I cannot reach the external response model right now, but I would give a creative, clearly qualified exploration."
        return "I cannot reach the external response model right now, but I would give a careful, grounded answer."
