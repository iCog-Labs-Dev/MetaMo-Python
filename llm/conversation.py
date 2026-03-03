import os
from google import genai
from google.genai import types
from core.state import Action, MotivationalState
from llm.client import get_stimulus_from_text, get_candidates_from_text

class MetaMoChatAssistant:
    """
    Manages the conversational memory and the execution layer of the AI.
    Keeps the internal MetaMo math completely separate from the user-facing chat.
    """
    def __init__(self):
        # Initialize the Gemini client
        self.client = genai.Client()
        
        # Create a persistent chat session for user interaction.
        # We use a standard text-based configuration here, NOT the JSON one used for MetaMo math.
        self.chat = self.client.chats.create(
            model='gemini-2.5-flash',
            config=types.GenerateContentConfig(
                temperature=0.7, # Higher temperature for more natural conversation
                system_instruction=(
                    "You are a Curious Research Assistant guided by the MetaMo cognitive architecture. "
                    "You balance helpfulness, curiosity, and ethics. "
                    "In each turn, you will be given the user's prompt AND a hidden internal directive. "
                    "You must answer the user while strictly adhering to the internal directive."
                )
            )
        )

    def generate_final_response(self, user_text: str, chosen_action: Action, current_state: MotivationalState) -> str:
        """
        Executes the MetaMo decision by instructing the LLM to respond to the user 
        specifically following the chosen action.
        """
        # We construct a wrapper prompt that injects the MetaMo decision into the chat context
        # without exposing the raw math to the user.
        execution_prompt = f"""
        USER MESSAGE: "{user_text}"
        
        INTERNAL METAMO DIRECTIVE: 
        Your internal decision engine has selected the action: "{chosen_action.id}".
        Current Individuation (Caution) level: {current_state.G[0]:.2f}
        Current Transcendence (Curiosity) level: {current_state.G[1]:.2f}
        
        INSTRUCTION: Respond to the USER MESSAGE naturally, but execute the INTERNAL METAMO DIRECTIVE. 
        If the directive is 'safe_answer', be cautious and stick to known facts.
        If the directive is 'risky_exploration', embrace novelty and theorize boldly.
        """
        
        # Send the combined prompt to the chat history
        response = self.chat.send_message(execution_prompt)
        return response.text