# MetaMo Integration Function

The integration point should be a reusable MetaMo cycle function, not the
interactive loop itself.

`applications/research_assistant.py` currently mixes three concerns:

- application IO: reading `input()` and printing responses
- application adapters: text-to-`Stimulus`, text-to-`Action`, final response generation
- MetaMo orchestration: appraisal, decision, consensus, peer simulation, and state blending

For an extendable use case, extract the orchestration into a function that
accepts prepared MetaMo inputs and returns a structured output. An interactive
loop, HTTP endpoint, robotics runtime, simulation, or batch job can then call
that function.

## Core Shape

The central function should do one motivational cycle:

```python
result = run_metamo_cycle(
    bimonad=bimonad,
    states=subsystem_states,
    stimulus=stimulus,
    candidates=candidates,
    consensus_pair=("curiosity", "ethics"),
    translator=translator,
)
```

It should not read from the terminal, call an LLM response generator, print
debug output, or mutate the caller's state dictionary. It should return the
chosen action, the consensus target state, and the updated subsystem states.

## Input Contract

The function should receive:

- `bimonad`: a `MetaMoPseudoBimonad(OpenPsiAppraisal(), MagusDecision())`
  instance, created once by the host application.
- `states`: a mapping from subsystem name to `MotivationalState`.
- `stimulus`: the current `Stimulus` already produced by the perception layer.
- `candidates`: the current list of candidate `Action` objects.
- `consensus_pair`: the two subsystem names to use for direct consensus.
- `translator`: optional `TranslationFunctor` for peer simulation.
- `blend`: whether to apply `blend_states()` before returning next states.

The required MetaMo data types are already in `core/state.py`:

```python
MotivationalState(G, M)
Stimulus(novelty, conduciveness, risk, effort)
Action(id, goal_correlations, risk_estimate, delta_g)
```

## Output Contract

Return one structured object with all values that a caller needs:

- `final_action`: the consensus action selected for execution.
- `merged_current`: the merged state before this cycle's transition.
- `merged_target`: the consensus target state after this cycle's transition.
- `next_states`: new subsystem states after local target blending.
- `local_actions`: each subsystem's preferred action before consensus.
- `local_targets`: each subsystem's local target state before blending.
- `peer_simulations`: optional translated peer-state views.

The caller can then execute `final_action`, show diagnostics from
`local_actions`, and replace its stored subsystem states with `next_states`.

## Suggested Implementation

This is the general portion to extract from `research_assistant.py`.

```python
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from category.bimonad import MetaMoPseudoBimonad
from category.functors import TranslationFunctor
from core.state import Action, MotivationalState, Stimulus
from dynamics.coherence import blend_states


@dataclass
class MetaMoCycleResult:
    final_action: Action
    merged_current: MotivationalState
    merged_target: MotivationalState
    next_states: dict[str, MotivationalState]
    local_actions: dict[str, Action]
    local_targets: dict[str, MotivationalState]
    peer_simulations: dict[str, MotivationalState] = field(default_factory=dict)


def merge_subsystem_states(
    bimonad: MetaMoPseudoBimonad,
    states: Mapping[str, MotivationalState],
) -> MotivationalState:
    names = list(states)
    if not names:
        raise ValueError("At least one motivational state is required.")

    merged = states[names[0]]
    for name in names[1:]:
        merged = bimonad.parallel_merge(merged, states[name])
    return merged


def run_metamo_cycle(
    *,
    bimonad: MetaMoPseudoBimonad,
    states: Mapping[str, MotivationalState],
    stimulus: Stimulus,
    candidates: Sequence[Action],
    consensus_pair: tuple[str, str] | None = None,
    translator: TranslationFunctor | None = None,
    blend: bool = True,
) -> MetaMoCycleResult:
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("At least one candidate action is required.")
    if not states:
        raise ValueError("At least one motivational state is required.")

    merged_current = merge_subsystem_states(bimonad, states)

    local_actions: dict[str, Action] = {}
    local_targets: dict[str, MotivationalState] = {}

    for name, state in states.items():
        action, target = bimonad.step(state, stimulus, candidate_list)
        local_actions[name] = action
        local_targets[name] = target

    names = list(states)
    if len(names) == 1:
        only_name = names[0]
        final_action = local_actions[only_name]
        merged_target = local_targets[only_name]
    else:
        if consensus_pair is None:
            if len(names) != 2:
                raise ValueError(
                    "consensus_pair is required when more than two states are supplied."
                )
            consensus_pair = (names[0], names[1])

        left_name, right_name = consensus_pair
        final_action, merged_target = bimonad.consensus_transition(
            states[left_name],
            states[right_name],
            stimulus,
            candidate_list,
        )

    next_states = {
        name: blend_states(state, local_targets[name]) if blend else local_targets[name]
        for name, state in states.items()
    }

    peer_simulations = {}
    if translator is not None:
        peer_simulations = {
            name: translator.simulate_peer(state)
            for name, state in states.items()
        }

    return MetaMoCycleResult(
        final_action=final_action,
        merged_current=merged_current,
        merged_target=merged_target,
        next_states=next_states,
        local_actions=local_actions,
        local_targets=local_targets,
        peer_simulations=peer_simulations,
    )
```

## How `interactive_loop()` Should Use It

The loop becomes a thin adapter. It owns user input, candidate generation, and
response execution. The MetaMo integration function owns the motivational
transition.

```python
import numpy as np

from category.bimonad import MetaMoPseudoBimonad
from category.functors import TranslationFunctor
from core.config import NUM_GOALS, NUM_MODULATORS, M_AROUSAL, M_SECURING
from magus.decision import MagusDecision
from openpsi.appraisal import OpenPsiAppraisal


def interactive_loop():
    bimonad = MetaMoPseudoBimonad(OpenPsiAppraisal(), MagusDecision())
    translator = TranslationFunctor(
        goal_translation=np.eye(NUM_GOALS),
        modulator_translation=np.eye(NUM_MODULATORS),
    )

    states = {
        "curiosity": create_curiosity_state(),
        "ethics": create_ethics_state(),
    }
    assistant = MetaMoChatAssistant()

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        stimulus = get_stimulus_from_text(user_input)

        merged_current = merge_subsystem_states(bimonad, states)
        current_mood = {
            "arousal": merged_current.M[M_AROUSAL],
            "caution": merged_current.M[M_SECURING],
        }
        candidates = get_candidates_from_text(user_input, current_mood)

        result = run_metamo_cycle(
            bimonad=bimonad,
            states=states,
            stimulus=stimulus,
            candidates=candidates,
            consensus_pair=("curiosity", "ethics"),
            translator=translator,
        )

        response_text = assistant.generate_final_response(
            user_input,
            result.final_action,
            result.merged_target,
        )

        states = result.next_states

        print(f"\nAssistant: {response_text}")
        print(
            "\n[Consensus State -> "
            f"Individuation: {result.merged_target.G[0]:.2f} | "
            f"Transcendence: {result.merged_target.G[1]:.2f}]"
        )
```

In this structure, `interactive_loop()` can be replaced by any host runtime.
The only required work for a new use case is to produce `Stimulus` and `Action`
inputs, then execute the returned `final_action`.

## What The Integration Function Owns

`run_metamo_cycle()` should integrate these MetaMo subsystems:

- `OpenPsiAppraisal` through `bimonad.step()`
- `MagusDecision` through `bimonad.step()` and `consensus_transition()`
- parallel subsystem evaluation through repeated `bimonad.step()` calls
- consensus action selection through `bimonad.consensus_transition()`
- optional reciprocal simulation through `TranslationFunctor`
- coherence-preserving state updates through `blend_states()`

It should not own:

- raw input collection
- prompt construction
- LLM calls
- UI or CLI printing
- robot control, API side effects, or workflow dispatch
- final natural-language response generation

Those are application adapters around the MetaMo cycle.

## Extension Points

### State profiles

Different applications can pass different subsystem states:

```python
states = {
    "task_completion": task_completion_state,
    "operator_safety": operator_safety_state,
}
```

The integration function does not need to know what those names mean. The names
only matter for selecting `consensus_pair`.

### Perception

Every use case must map raw input to:

```python
Stimulus(
    novelty=...,
    conduciveness=...,
    risk=...,
    effort=...,
)
```

For the research assistant, this comes from `get_stimulus_from_text()`. Another
system could use a sensor model, event classifier, rule engine, or direct
numeric input.

### Candidate generation

Every use case must provide candidate actions:

```python
Action(
    id="domain_action_name",
    goal_correlations=np.array([...], dtype=float),
    risk_estimate=...,
    delta_g=np.array([...], dtype=float),
)
```

Action ids are domain commands. `MagusDecision` scores the numeric fields, not
the text meaning of the id.

### Execution

The caller decides how to execute the returned `final_action`. For example:

- a chat app turns it into a response instruction
- a robot maps it to a controller command
- a workflow system maps it to a job transition
- a simulation records it as the next agent move

The integration function should stop at returning the selected action and
updated motivational states.

## Research Assistant Specific Pieces To Keep Outside

These are application-specific and should remain outside `run_metamo_cycle()`:

- `get_stimulus_from_text(user_input)`
- `get_candidates_from_text(user_input, current_mood)`
- `MetaMoChatAssistant.generate_final_response(...)`
- chat-specific action ids in `llm/action_schema.py`
- CLI printing of subsystem decisions

The reusable insight from `research_assistant.py` is the sequence of MetaMo
operations, not the chat interface around it.
