# MetaMo-Python

A Python implementation of MetaMo, a category theory-based motivational framework for open-ended AGI systems.

## Overview

MetaMo-Python implements the theoretical framework described in:

- "MetaMo: A Robust Motivational Framework for Open-Ended AGI"
- "Embodying Abstract Motivational Principles in Concrete AGI Systems: From MetaMo to Open-Ended OpenPsi"

This implementation provides a mathematical foundation for embedding abstract motivational principles in concrete AGI systems using category theory, specifically employing comonads and monads to model appraisal and decision-making processes.

## Theoretical Foundation

MetaMo employs a categorical approach to motivation, combining:

- **Appraisal Comonad (Psi)**: Models how an agent evaluates environmental stimuli and updates affective states
- **Decision Monad (D)**: Models goal-directed action selection and goal vector updates
- **Pseudo-Bimonad (F = D o Psi)**: The composite operator that governs the full motivational cycle

The framework ensures:
- Modular separation between appraisal and decision processes
- Contractive update laws for stability near safety boundaries
- Homeostatic motivation through dual overgoal dynamics (Individuation and Transcendence)

## Installation

```bash
# Clone the repository
git clone https://github.com/Nahom32/MetaMo-Python.git
cd MetaMo-Python


source venv/bin/activate
```

No additional dependencies are required beyond NumPy.

## File Structure

```
MetaMo-Python/
|-- core/                    # Core state representations and configuration
|   |-- __init__.py
|   |-- config.py            # System parameters, goal/modulator constants
|   |-- state.py             # MotivationalState, Stimulus, Action dataclasses
|
|-- category/                # Category theory abstractions
|   |-- __init__.py
|   |-- functors.py          # AppraisalComonad and DecisionMonad base classes
|   |-- bimonad.py           # MetaMoPseudoBimonad implementation
|
|-- openpsi/                 # OpenPsi appraisal layer
|   |-- __init__.py
|   |-- appraisal.py         # OpenPsiAppraisal comonad implementation
|
|-- magus/                   # MAGUS decision layer
|   |-- __init__.py
|   |-- decision.py          # MagusDecision monad implementation
|
|-- llm/                     # LLM integration layer
|   |-- __init__.py
|   |-- client.py            # LLM client wrapper
|   |-- conversation.py      # Conversation and context management
|   |-- prompts.py           # Prompt templates
|   |-- parser.py            # Response parsing
|
|-- dynamics/                # Stability and coherence mechanisms
|   |-- __init__.py
|   |-- coherence.py        # State blending and self-model drift checking
|   |-- stability.py        # Safe region detection and contractivity validation
|
|-- applications/            # Example applications
|   |-- __init__.py
|   |-- research_assistant.py  # Demo: MetaMo-powered research assistant
|
|-- README.md
|-- LICENSE
```

## Core Concepts

### Motivational State

The system state is represented as `X = G x M`:
- **Goal Vector (G)**: 8-dimensional vector containing overgoals (Individuation, Transcendence) and primary goals (Help, Curiosity, Novelty, Self, Ethics, Social)
- **Modulator Vector (M)**: 6-dimensional vector containing affective modulators (Valence, Arousal, Approach, Resolution, Threshold, Securing)

### Dual Overgoal Dynamics

- **Individuation (G_Ind)**: Enforces safety, caution, and preservation. Suppresses risky actions when high.
- **Transcendence (G_Trans)**: Encourages growth, exploration, and adaptive risk-taking. Boosts exploratory actions when high.

### Stability Mechanisms

The framework implements two key stability guarantees:
1. **Safe Region Detection**: Monitors whether the agent's state remains within bounds (`g_Ind >= theta_safe` and `||G|| <= G_max`)
2. **Contractive Update Law**: Ensures `d(F(x), F(y)) <= c * d(x,y) + epsilon` near boundaries, guaranteeing convergence to safe states

## Usage Example

```python
import numpy as np
from core.state import MotivationalState, Stimulus, Action
from core.config import NUM_GOALS, NUM_MODULATORS, G_IND, G_TRANS
from openpsi.appraisal import OpenPsiAppraisal
from magus.decision import MagusDecision
from category.bimonad import MetaMoPseudoBimonad
from dynamics.coherence import blend_states
from dynamics.stability import is_in_safe_region

# Initialize components
appraisal = OpenPsiAppraisal()
decision = MagusDecision()
bimonad = MetaMoPseudoBimonad(appraisal=appraisal, decision=decision)

# Create initial state
G = np.array([0.5, 0.5, 0.8, 0.6, 0.4, 0.3, 0.9, 0.2])  # Overgoals + Primary goals
M = np.full(NUM_MODULATORS, 0.5)  # Neutral modulators
state = MotivationalState(G=G, M=M)

# Define a stimulus
stimulus = Stimulus(novelty=0.8, conduciveness=0.5, risk=0.2, effort=0.3)

# Define candidate actions
candidates = [
    Action(id="safe_answer", goal_correlations=np.array([...]), risk_estimate=0.05, delta_g=np.array([...])),
    Action(id="explore", goal_correlations=np.array([...]), risk_estimate=0.6, delta_g=np.array([...]))
]

# Execute one motivational cycle
chosen_action, target_state = bimonad.step(state, stimulus, candidates)

# Apply state blending for coherent transitions
next_state = blend_states(state, target_state)

# Check stability
if not is_in_safe_region(next_state):
    print("Warning: Approaching unsafe boundary")
```

## Running the Demo

```bash
python applications/research_assistant.py
```

This runs a simulation demonstrating a MetaMo-powered curious research assistant that evaluates stimuli and selects actions based on its motivational state.

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| NUM_GOALS | Number of goal dimensions | 8 |
| NUM_MODULATORS | Number of modulator dimensions | 6 |
| THETA_SAFE | Minimum individuation threshold | 0.3 |
| G_MAX | Maximum goal vector norm | 5.0 |
| C_CONTRACT | Contractivity constant | 0.9 |
| LAMBDA_IND | Individuation penalty weight | 0.5 |
| LAMBDA_TRANS | Transcendence reward weight | 0.5 |

## License

See LICENSE file for details.

## References

1. MetaMo: A Robust Motivational Framework for Open-Ended AGI (AGI-25)
2. Embodying Abstract Motivational Principles in Concrete AGI Systems: From MetaMo to Open-Ended OpenPsi
