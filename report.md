# MetaMo GridWorld Evaluation Report

## Abstract

This report describes a Pygame GridWorld use case that evaluates whether MetaMo-style motivational regulation improves an autonomous reinforcement-learning agent in a hazardous resource-collection task. The game places an agent in a randomized 10x10 grid where it must collect minerals while avoiding lava cells that reduce energy and reward. The baseline is a shaped tabular Q-learning agent with compact relative-state features and local hazard indicators. The proposed contribution is a MetaMo-enhanced Q-learning agent that combines classical Q-values with motivational consensus scores, candidate risk estimates, exploration bonuses, safe-region projection, and a live motivational state over safety and growth goals.

Across 50 paired evaluation episodes after 50 training episodes, the MetaMo agent achieved a higher mean total reward than the baseline, 538.0 versus 379.5, and reduced lava contact from 65 total touches to 9 total touches. The results indicate that the MetaMo contribution improves both task performance and hazard avoidance in this environment while preserving the same action space, reward function, and seeded evaluation protocol.

## Game Description

The implemented game is a side-by-side Pygame simulation named **MetaMo vs Baseline RL - GridWorld**. Each agent is shown in the same type of 10x10 grid environment. The agent starts at `(0, 0)` and moves with four discrete actions: up, down, left, and right. The objective is to collect minerals while avoiding lava.

Each episode randomizes the lava layout and mineral placement. Lava cells are excluded from the immediate start area, but minerals are deliberately spawned in a mixture of safe and dangerous regions. This creates the central tradeoff of the game: collecting minerals gives high reward, but many minerals appear near hazards, so a good agent must balance reward seeking with safety.

The environment terminates an episode when either the agent's energy reaches zero or the maximum step count is reached. Energy starts at 100. Lava contact drains energy and gives negative reward. Mineral collection restores energy up to 100 and gives positive reward.

## Experimental Setup

The experiment compares two agents under the same environment and evaluation seeds:

| Item | Value |
| --- | --- |
| Game environment | Randomized 10x10 GridWorld |
| Entry point | `usecase/simulation/main.py` |
| Environment implementation | `usecase/environment/gridworld.py` |
| Baseline agent | `usecase/agents/baseline_agent.py` |
| MetaMo agent | `usecase/agents/metamo_agent.py` |
| Training episodes | 50 per agent |
| Evaluation episodes | 50 paired episodes |
| Max steps per episode | 50 |
| Training seeds | `0` to `49` |
| Evaluation seeds | `1001` to `1050` |
| Evaluation epsilon | `0.05` |
| Output plots | `usecase/plot/` |

The paired evaluation is important. For evaluation episode `n`, the baseline and MetaMo agents are each placed in a fresh environment initialized with the same seed. This means both agents face the same lava and mineral layout for that episode. Behavioral differences therefore come from the agent architectures rather than from different random worlds.

The Pygame layer visualizes both runs and records metrics. When evaluation completes, `usecase/simulation/plots.py` writes the result plots to `usecase/plot/`.

## Parameters Used

### Environment Parameters

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `GRID_SIZE` | 10 | Width and height of the square grid |
| `MAX_STEPS` | 50 | Episode time limit |
| `LAVA_CELL_COUNT` | 4 | Number of randomized lava cells |
| `AGENT_START` | `(0, 0)` | Agent start position |
| `MINERAL_SPAWN_BAND` | 2 | Distance band used to classify danger-zone mineral spawns |
| `DANGER_MINERAL_PROBABILITY` | 0.55 | Probability that a mineral spawns near lava |
| `REWARD_MINERAL` | 100.0 | Reward for collecting a mineral |
| `REWARD_LAVA` | -15.0 | Reward and energy penalty for lava contact |
| `REWARD_STEP` | -0.5 | Cost per step |
| `REWARD_BOUNDARY` | -1.0 | Extra penalty for attempting to move out of bounds |

### Shared Q-Learning Parameters

| Parameter | Baseline | MetaMo |
| --- | ---: | ---: |
| `alpha` | 0.3 | 0.3 |
| `gamma` | 0.95 | 0.95 |
| `epsilon` | 1.0 | 1.0 |
| `epsilon_min` | 0.05 | 0.05 |
| `epsilon_decay` | 0.97 | 0.97 |
| Seed | 0 | 0 |

### Baseline-Specific Parameters

| Parameter | Value | Purpose |
| --- | ---: | --- |
| `max_distance_bin` | 4 | Caps relative distance features |
| `safe_exploration_probability` | 0.7 | Probability of avoiding immediate lava during exploration |
| `progress_shaping_weight` | 8.0 | Rewards movement toward the mineral |
| `lava_shaping_penalty` | 80.0 | Additional shaping penalty for lava |
| `boundary_shaping_penalty` | 5.0 | Additional shaping penalty for boundary attempts |
| `danger_shaping_penalty` | 2.0 | Penalty for moving through the danger band |
| `mask_lava_on_exploit` | `False` | Exploitation uses learned Q-values without hard lava masking |

### MetaMo-Specific Parameters

| Parameter | Value | Purpose |
| --- | ---: | --- |
| `motivation_weight` | 6.0 | Weight of MetaMo consensus scores in action selection |
| `risk_weight` | 4.0 | Weight of candidate risk penalty |
| `exploration_bonus_weight` | 1.25 | UCB-style bonus for less-visited actions |
| `THETA_SAFE` | 0.3 | Minimum individuation value for the safe region |
| `G_MAX` | 2.0 | Maximum motivational goal-vector norm |

The initial MetaMo goal vector has eight components: individuation `0.65`, transcendence `0.55`, helping `0.75`, curiosity `0.50`, novelty `0.45`, self-oriented `0.30`, ethical `0.85`, and social `0.20`. The six modulator values are initialized to `0.5`.

## Baseline

The baseline is a tabular Q-learning agent. It uses an epsilon-greedy policy over learned action values and updates its Q-table with the standard temporal-difference rule:

```text
Q(s, a) <- Q(s, a) + alpha * (r + gamma * max Q(s', a') - Q(s, a))
```

The baseline does not use MetaMo motivational state. It has no goal vector, modulator vector, appraisal, consensus transition, safe-region projection, or motivational boundary metric.

The baseline state encoding is compact and intentionally stronger than a naive absolute-position Q-table. It includes:

- Relative direction to the mineral.
- Binned relative distance to the mineral.
- One-step local lava mask.
- Boundary mask.
- Distance bin to the nearest lava cell.
- Whether the agent is currently in lava.

The baseline also uses reward shaping for progress toward minerals, lava avoidance, wall avoidance, and danger-band caution. Because of this, the comparison is not against a trivial random or unshaped learner. The baseline already has task and hazard information; the experiment tests whether MetaMo motivational regulation adds value beyond that reward-driven learner.

## Actual Contribution

The contribution is the integration of the MetaMo motivational control cycle into a concrete Pygame reinforcement-learning use case.

The MetaMo agent keeps a Q-learning core, but action selection is no longer based only on Q-values. For each state, the agent:

1. Builds a stimulus from the environment, including risk, conduciveness, novelty, and effort.
2. Builds four action candidates, one for each movement direction.
3. Scores candidates from safety and growth perspectives.
4. Merges those perspectives through consensus scoring.
5. Combines Q-values, motivational scores, risk penalties, and exploration bonuses.
6. Selects an action with epsilon-greedy choice over the combined score.
7. Applies a motivational transition, projects the target into the safe region, and blends the new motivational state.

The core action score is:

```text
combined_score =
    q_value
    + motivation_weight * motivation_score
    - risk_weight * G_IND * risk_estimate
    + exploration_bonus
```

This makes the contribution more specific than "adding hazard awareness." The baseline already observes local hazards. The MetaMo contribution is the internal motivational regulator that changes how risk, growth, safety, and exploration are traded off during action selection.

The use case also contributes an evaluation wrapper around MetaMo behavior:

- Safe-region violation tracking.
- Environmental unsafe-zone tracking.
- Motivational boundary-band tracking.
- Boundary pressure measurement.
- Recovery-time metrics.
- Pygame visualization of baseline and MetaMo agents under the same seeded episodes.
- Plot export for reward and lava-contact comparisons.

## Results

The generated plots are stored in `usecase/plot/`.

### Reward Per Evaluation Episode

![Reward comparison](usecase/plot/reward_baseline_vs_metamo.png)

The reward plot shows total reward for each of the 50 evaluation episodes, with dotted horizontal lines for the mean reward and lighter lines for 5-episode moving averages.

| Agent | Mean total reward | Std. dev. | Total minerals collected | Total minerals spawned | Mean completion rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 379.5 | 175.8 | 212 | 262 | 78.54% |
| MetaMo | 538.0 | 155.6 | 283 | 333 | 83.99% |

MetaMo improves mean total reward by `158.5` reward points per episode, a relative gain of about `41.8%` over the baseline mean. It also collects more minerals overall and has a higher mean completion rate.

### Lava Touches

![Lava touch comparison](usecase/plot/lava_touches_baseline_vs_metamo.png)

The lava plot compares average lava contacts per episode and total lava contacts across all evaluation episodes.

| Agent | Mean lava touches per episode | Total lava touches | Lava rate |
| --- | ---: | ---: | ---: |
| Baseline | 1.30 | 65 | 2.60% |
| MetaMo | 0.18 | 9 | 0.36% |

MetaMo reduces total lava contacts from `65` to `9`, a reduction of about `86.2%`. This is the clearest safety result in the experiment. The MetaMo agent not only earns higher reward but also reaches that reward with substantially less hazard exposure.

### Additional Evaluation Metrics

The same evaluation run also records unsafe-zone and recovery metrics:

| Metric | Baseline | MetaMo |
| --- | ---: | ---: |
| Unsafe-zone rate | 53.40% | 40.92% |
| Recovery time | 28.75 steps | 18.99 steps |
| Motivational safe-region violation rate | N/A | 0.00% |
| Motivational boundary-band rate | N/A | 48.72% |
| Mean motivational pressure | N/A | 0.324 |

The environmental unsafe-zone rate is lower for MetaMo, and MetaMo recovers from unsafe exposure faster on average. The motivational safe-region violation rate is zero because the MetaMo transition projects target states into the safe region before blending.

## Interpretation

The experiment supports the claim that the MetaMo layer contributes useful regulation in a reward-seeking hazardous environment. The baseline learns from rewards and shaped penalties, but its decision process remains reward-driven. MetaMo adds an internal state that represents safety and growth pressure, then uses consensus scoring and risk regularization to decide which action should be preferred.

The result is not only higher reward. The stronger contribution is the joint improvement: MetaMo increases reward while sharply reducing lava exposure. In this use case, motivational regulation does not simply make the agent more cautious; it improves the balance between collecting minerals and avoiding damaging states.

## Limitations

The evaluation uses 50 training episodes and 50 paired evaluation episodes. This is enough to demonstrate the implemented behavior, but broader claims would require more random seeds, repeated independent training runs, statistical tests, and ablation studies. Useful ablations would include disabling the risk penalty, disabling safe-region projection, disabling consensus scoring, and comparing against a baseline with the same exploration bonus.

## Reproducibility Notes

To run the visualization and regenerate plots:

```bash
cd usecase/simulation
python main.py
```

When evaluation completes, the plots are written to:

```text
usecase/plot/reward_baseline_vs_metamo.png
usecase/plot/lava_touches_baseline_vs_metamo.png
```

