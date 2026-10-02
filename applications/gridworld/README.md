# GridWorld

GridWorld is a rule-based MetaMo application with a tabular Q-learning
baseline. It contains the environment, task-specific motivational model,
agents, controlled evaluation tools, and an optional visual simulation.

```text
applications/gridworld/
|-- agents/        baseline and MetaMo-guided Q-learning agents
|-- assets/        simulation sprites and audio
|-- evaluation/    experiment runners, metrics, plots, and smoke checks
|-- simulation/    optional pygame visualization
|-- config.py      environment and calibration constants
|-- environment.py GridWorld environment
|-- schema.py      goal and modulator coordinates
|-- stimulus.py    normalized appraisal input
|-- profile.py     task-specific appraisal and action semantics
|-- runtime.py     binding to the existing MetaMo pipeline
`-- state.py       initial motivational state
```

## Variables

The two MetaMo overgoals remain `individuation` and `transcendence`.

The application uses three primary goals:

| Goal | Initial value | Operational meaning |
|---|---:|---|
| `mineral_acquisition` | 0.75 | make progress toward and collect minerals |
| `energy_preservation` | 0.65 | avoid energy loss and recover energy |
| `navigation_efficiency` | 0.45 | move productively without boundary waste |

Two anti-goals make hazards explicit:

| Anti-goal | Initial value | Activated by |
|---|---:|---|
| `lava_exposure` | 0.85 | local lava risk |
| `boundary_collision` | 0.35 | an action that tries to leave the grid |

The appraisal stimulus contains six normalized signals:

| Signal | Meaning |
|---|---|
| `goal_proximity` | closeness to the current mineral |
| `hazard_pressure` | calibrated pressure from nearby lava |
| `energy_deficit` | missing energy as a fraction of full energy |
| `time_pressure` | fraction of the episode budget already used |
| `safe_progress` | mineral proximity discounted by nearby hazard |
| `safe_mobility` | fraction of directions that stay in bounds and avoid lava |

## Runtime process

For every GridWorld step:

1. The application builds six appraisal signals and four action candidates.
2. Rule appraisal updates motivation through MetaMo's existing goal-update,
   projection, coherence, law-audit, and safety pipeline.
3. Task and safety perspectives score the actions.
4. Q-learning supplies a near-optimal action shortlist.
5. The composed selector chooses within the shortlist and rejects immediate
   lava entry when a safer legal action exists.
6. Q updates occur during training only. Evaluation freezes Q while preserving
   MetaMo's motivational transition.

## Evaluation variants

The default evaluation compares:

| Variant | Purpose |
|---|---|
| `BaselineCompactQ` | Q-learning control |
| `MetaMoTaskSelector` | task perspective only |
| `MetaMoSafetySelector` | safety perspective only |
| `MetaMoComposedSelector` | composed task and safety perspectives |
| `MetaMo` | composed selector with hard immediate-lava safety |

### Selector-withdrawal experiment

The selector-withdrawal experiment separates improvement learned into the
Q-table from improvement supplied by MetaMo while actions are being chosen:

| Variant | Training policy | Evaluation policy | Question |
|---|---|---|---|
| `QTrainQEval` | Q only | Q only | baseline |
| `MetaMoTrainQEval` | MetaMo-guided | Q only | did guided experience improve Q itself? |
| `QTrainMetaMoEval` | Q only | MetaMo | what does MetaMo add only at decision time? |
| `MetaMoTrainMetaMoEval` | MetaMo-guided | MetaMo | total combined effect |

## Run

Run the lightweight validation:

```powershell
python -m applications.gridworld.evaluation.smoke_checks
```

Run and plot the default controlled evaluation:

```powershell
python -m applications.gridworld.evaluation.runner `
  --output-dir eval_results/gridworld

python -m applications.gridworld.evaluation.analyze_ablation `
  --input-dir eval_results/gridworld

python -m applications.gridworld.evaluation.plot_results `
  --input-dir eval_results/gridworld
```

Run the standalone-Q learning-transfer diagnostic:

```powershell
python -m applications.gridworld.evaluation.learning_transfer --quiet-seeds
```

Run the visual simulation after installing `pygame`:

```powershell
python -m applications.gridworld.simulation.main
```

The non-visual application requires NumPy. Generated results are written under
`eval_results/` and are not application source.
