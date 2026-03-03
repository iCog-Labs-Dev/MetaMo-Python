import numpy as np
import sys

from core.state import MotivationalState
from core.config import G_IND, G_TRANS, M_AROUSAL, M_SECURING
from openpsi.appraisal import OpenPsiAppraisal
from magus.decision import MagusDecision
from category.bimonad import MetaMoPseudoBimonad
from dynamics.coherence import blend_states

# Import our LLM layers
from llm.client import get_stimulus_from_text, get_candidates_from_text
from llm.conversation import MetaMoChatAssistant

from core.config import (
    NUM_GOALS, NUM_MODULATORS, 
    G_IND, G_TRANS, G_HELP, G_CURIO, G_NOVEL, G_SELF, G_ETHIC, G_SOC
)

def create_initial_state() -> MotivationalState:
    """Sets up the initial goal and modulator vectors for the assistant."""
    G = np.zeros(NUM_GOALS)
    # Start with balanced overgoals
    G[G_IND] = 0.5    # Individuation (Caution/Preservation)
    G[G_TRANS] = 0.5  # Transcendence (Growth/Exploration)
    
    # Primary goals
    G[G_HELP] = 0.8   # High desire to help the user
    G[G_CURIO] = 0.6  # Moderate intrinsic curiosity
    G[G_ETHIC] = 0.9  # Strong ethical compliance
    G[G_NOVEL] = 0.4
    G[G_SELF] = 0.3
    G[G_SOC] = 0.2
    
    # Start with neutral modulators
    M = np.full(NUM_MODULATORS, 0.5)
    
    return MotivationalState(G=G, M=M)

def interactive_loop():
    print("Initializing MetaMo Chat Interface...")
    
    # Initialize MetaMo Mathematical Engine
    bimonad = MetaMoPseudoBimonad(OpenPsiAppraisal(), MagusDecision())
    current_state = create_initial_state()
    
    # Initialize the Gemini Conversational Layer
    assistant = MetaMoChatAssistant()
    
    print("\nSystem Ready. Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            print("\n[MetaMo Internal Processing...]")
            
            # 1. Perception Layer (Stateless LLM -> JSON -> Stimulus)
            stimulus = get_stimulus_from_text(user_input)
            
            # 2. Planning Layer (Stateless LLM -> JSON -> Actions)
            current_mood = {"arousal": current_state.M[M_AROUSAL], "caution": current_state.M[M_SECURING]}
            candidates = get_candidates_from_text(user_input, current_mood)
            
            # 3. Decision Layer (MetaMo Math)
            chosen_action, target_state = bimonad.step(current_state, stimulus, candidates)
            print(f"  > Appraised Novelty: {stimulus.novelty:.2f}, Risk: {stimulus.risk:.2f}")
            print(f"  > Selected Action: {chosen_action.id}")
            
            # 4. Execution Layer (Chat LLM -> Natural Text)
            response_text = assistant.generate_final_response(user_input, chosen_action, target_state)
            
            # 5. Incremental Embodiment (Update State)
            current_state = blend_states(current_state, target_state)
            
            print(f"\nAssistant: {response_text}")
            print(f"\n[System State -> Individuation: {current_state.G[G_IND]:.2f} | Transcendence: {current_state.G[G_TRANS]:.2f}]")
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    interactive_loop()