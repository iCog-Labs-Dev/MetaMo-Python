import numpy as np
import os
import sys

if __package__ in (None, ""):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from category.functors import TranslationFunctor
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
    print("Initializing MetaMo Multi-Subsystem Chat Interface...")
    
    # Initialize the core MetaMo mathematical engine
    bimonad = MetaMoPseudoBimonad(OpenPsiAppraisal(), MagusDecision())
    assistant = MetaMoChatAssistant()
    
    # Initialize two parallel subsystems 
    state_curiosity = create_initial_state()
    state_curiosity.G[G_TRANS] = 0.9 # Highly transcendent/curious
    
    state_ethics = create_initial_state()
    state_ethics.G[G_IND] = 0.9      # Highly individuated/cautious
    
    # Initialize a Translation Functor for peer simulation (Identity matrix for simplicity)
    identity_matrix = np.eye(NUM_GOALS)
    translator = TranslationFunctor(identity_matrix)
    
    print("\nSystem Ready. Subsystems: [Curiosity] & [Ethics]. Type 'quit' to exit.")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            print("\n[MetaMo Internal Processing...]")
            
            # 1. Perception Layer
            stimulus = get_stimulus_from_text(user_input)
            
            # 2. Planning Layer (Use merged mood for generating candidates)
            # CALLING THE METHOD FROM THE BIMONAD INSTANCE
            merged_current = bimonad.parallel_merge(state_curiosity, state_ethics)
            current_mood = {"arousal": merged_current.M[M_AROUSAL], "caution": merged_current.M[M_SECURING]}
            candidates = get_candidates_from_text(user_input, current_mood)
            
            # 3. Parallel Decision Layer 
            action_c, target_c = bimonad.step(state_curiosity, stimulus, candidates)
            action_e, target_e = bimonad.step(state_ethics, stimulus, candidates)
            
            print(f"  > [Curiosity Subsystem] wants to: {action_c.id}")
            print(f"  > [Ethics Subsystem] wants to: {action_e.id}")
            
            # 4. Peer Simulation 
            simulated_ethics = translator.simulate_peer(state_curiosity)
            print(f"  > [Reciprocal Simulation]: Curiosity agent predicts Ethics agent's caution is {simulated_ethics.G[G_IND]:.2f}")

            # 5. Parallel Merge 
            # CALLING THE METHOD FROM THE BIMONAD INSTANCE
            merged_target = bimonad.parallel_merge(target_c, target_e)
            
            # Ensure the merged action respects the consensus 
            final_action = action_e if action_e.risk_estimate < action_c.risk_estimate else action_c
            
            # 6. Execution Layer
            response_text = assistant.generate_final_response(user_input, final_action, merged_target)
            
            # 7. Incremental Embodiment (Update States)
            state_curiosity = blend_states(state_curiosity, target_c)
            state_ethics = blend_states(state_ethics, target_e)
            
            print(f"\nAssistant: {response_text}")
            print(f"\n[Consensus State -> Individuation: {merged_target.G[G_IND]:.2f} | Transcendence: {merged_target.G[G_TRANS]:.2f}]")
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    interactive_loop()