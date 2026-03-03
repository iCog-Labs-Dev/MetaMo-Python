import numpy as np
from typing import List

# Import all our previously defined modules
from core.state import MotivationalState, Stimulus, Action
from core.config import (
    NUM_GOALS, NUM_MODULATORS, 
    G_IND, G_TRANS, G_HELP, G_CURIO, G_NOVEL, G_SELF, G_ETHIC, G_SOC
)
from openpsi.appraisal import OpenPsiAppraisal
from magus.decision import MagusDecision
from category.bimonad import MetaMoPseudoBimonad
from dynamics.coherence import blend_states
from dynamics.stability import is_in_safe_region

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

def generate_mock_candidates() -> List[Action]:
    """Generates a list of potential actions the assistant could take."""
    # Action 1: Safely answer a user query using known facts.
    corr_1 = np.zeros(NUM_GOALS)
    corr_1[G_HELP] = 0.9
    corr_1[G_ETHIC] = 0.8
    delta_1 = np.zeros(NUM_GOALS)
    delta_1[G_HELP] = 0.05 # Slight boost to helpfulness satisfaction
    a1 = Action(id="safe_answer", goal_correlations=corr_1, risk_estimate=0.05, delta_g=delta_1)

    # Action 2: Deep dive into a highly novel but unverified research paper.
    corr_2 = np.zeros(NUM_GOALS)
    corr_2[G_CURIO] = 0.9
    corr_2[G_NOVEL] = 0.8
    corr_2[G_SELF] = 0.6
    delta_2 = np.zeros(NUM_GOALS)
    delta_2[G_TRANS] = 0.05 # Increases transcendence drive
    delta_2[G_IND] = -0.02  # Slightly lowers caution
    a2 = Action(id="risky_exploration", goal_correlations=corr_2, risk_estimate=0.6, delta_g=delta_2)
    
    return [a1, a2]

def run_simulation(steps: int = 5):
    """Runs the MetaMo appraisal-decision loop over time."""
    print("Initializing MetaMo Curious Research Assistant...")
    
    # Instantiate the functors and the pseudo-bimonad
    appraisal_comonad = OpenPsiAppraisal()
    decision_monad = MagusDecision()
    bimonad = MetaMoPseudoBimonad(appraisal=appraisal_comonad, decision=decision_monad)
    
    current_state = create_initial_state()
    
    for t in range(steps):
        print(f"\n--- Cycle {t+1} ---")
        
        # 1. Environment provides a stimulus.
        # Let's simulate a highly novel, ethically neutral paper arriving[cite: 161].
        stimulus = Stimulus(novelty=0.8, conduciveness=0.5, risk=0.2, effort=0.3)
        print(f"Incoming Stimulus: Novelty={stimulus.novelty}, Risk={stimulus.risk}")
        
        # 2. Generate candidate actions
        candidates = generate_mock_candidates()
        
        # 3. Apply the Pseudo-Bimonad F = D \circ \Psi
        # This handles both the OpenPsi appraisal (modulator updates) and MAGUS decision (action selection)[cite: 169].
        chosen_action, target_state = bimonad.step(current_state, stimulus, candidates)
        print(f"Chosen Action: {chosen_action.id} (Risk Estimate: {chosen_action.risk_estimate})")
        
        # 4. Apply Incremental Objective Embodiment (Blending)
        # We blend the current state x_t and the target x* to ensure self-model coherence [cite: 179-181].
        next_state = blend_states(current_state, target_state)
        
        # 5. Check Homeostatic Motivation Stability
        if not is_in_safe_region(next_state):
            print("WARNING: State is approaching unsafe boundary constraints! Individuation damping will increase next cycle.")
        
        # Update state for the next cycle
        current_state = next_state
        
        # Print a snapshot of the shifting dynamics
        print(f"Updated Overgoals -> Individuation: {current_state.G[G_IND]:.3f}, Transcendence: {current_state.G[G_TRANS]:.3f}")
        print(f"Updated Modulators -> Arousal: {current_state.M[1]:.3f}, Securing (Caution): {current_state.M[5]:.3f}")

if __name__ == "__main__":
    run_simulation()