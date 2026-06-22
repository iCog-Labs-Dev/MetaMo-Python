# MetaMo GridWorld — Integration Documentation

A side-by-side pygame simulation that compares a **Baseline Q-learning agent** against a
**MetaMo-enhanced Q-learning agent** in a randomized 10×10 hazard gridworld. Both agents
train for 50 episodes, then evaluate for 50 episodes simultaneously in the same environment
(identical seeds), so every difference in behaviour is caused by the architecture, not luck.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Setup & Running](#2-setup--running)
3. [Environment — GridWorld](#3-environment--gridworld)
4. [Baseline Agent — Tabular Q-Learning](#4-baseline-agent--tabular-q-learning)
5. [MetaMo Agent — Motivated Q-Learning](#5-metamo-agent--motivated-q-learning)
6. [Motivational State](#6-motivational-state)
7. [MetaMo Core Pipeline](#7-metamo-core-pipeline)
8. [How `select_action` Works in the MetaMo Agent](#8-how-select_action-works-in-the-metamo-agent)
9. [Metrics & Collector](#9-metrics--collector)
10. [Simulation Layer](#10-simulation-layer)
11. [Hyperparameter Reference](#11-hyperparameter-reference)

---

## 1. Project Structure

```
usecase/
├── agents/
│   ├── baseline_agent.py       ← Tabular Q-learning, no motivational layer
│   └── metamo_agent.py         ← Q-learning + MetaMo motivational regulation
│
├── assets/
│   
├── environment/
│   └── gridworld.py            ← 10×10 GridWorld (randomized lava, mixed mineral zones)
│
├── metamo/
│   ├── core.py                 ← Adapter: stimulus, candidates, consensus, transition
│   └── state.py                ← create_initial_motivational_state() for this usecase
│
├── metrics/
│   └── collector.py            ← EpisodeLog, MetricsCollector, SRV, Recovery Time
│
├── simulation/
│   ├── main.py                 ← Pygame event loop — run this
│   ├── renderer.py             ← draw_grid(), draw_panel(), all drawing logic
│   └── runner.py               ← train_agent(), step_baseline(), step_metamo()
│
└── README.md
```

**Entry point:** `python usecase/simulation/main.py`

---

## 2. Setup & Running

```bash
python -m venv venv
source venv/bin/activate         
pip install -r requirements.txt

cd usecase/simulation
python main.py
```

### Controls

| Key        | Action                        |
|------------|-------------------------------|
| `SPACE`    | Pause / Resume                |
| `R`        | Reset evaluation (full re-run)|
| `F / UP`   | Increase simulation speed     |
| `S / DOWN` | Decrease simulation speed     |
| `Q / ESC`  | Quit                          |

---

## 3. Environment — GridWorld

`environment/gridworld.py` is the world both agents share.

```
(0,0) — Agent always starts here
  │
  └── 10×10 grid
       ├── 4 lava cells  (randomized each episode, never within L1=2 of start)
       ├── Minerals spawn 55% near lava, 45% in safe cells
       └── Episode ends when energy ≤ 0 OR step_count ≥ MAX_STEPS
```

### Reward table

| Event             | Reward  | Purpose                              |
|-------------------|---------|--------------------------------------|
| Collect mineral   | `+100`  | Primary incentive                    |
| Step into lava    | `−15`   | Hazard penalty (energy drain too)    |
| Each step taken   | `−0.5`  | Time pressure — don't wander         |
| Boundary attempt  | `−1`    | Penalise walking into walls          |

Energy starts at 100. Collecting a mineral restores energy up to 100; lava drains it by 15.

### State dictionary returned to agents

| Key             | Type              | Meaning                                   |
|-----------------|-------------------|-------------------------------------------|
| `pos`           | `(int, int)`      | Agent (row, col)                          |
| `mineral_pos`   | `(int, int)`      | Mineral (row, col)                        |
| `energy`        | `float`           | Current energy (0–100)                    |
| `in_lava`       | `bool`            | Whether agent is currently on lava        |
| `lava_distance` | `int`             | Manhattan distance to nearest lava cell   |
| `lava_cells`    | `tuple`           | All lava positions this episode           |
| `dx_mineral`    | `int`             | Horizontal offset to mineral              |
| `dy_mineral`    | `int`             | Vertical offset to mineral                |
| `step`          | `int`             | Current step count                        |

The baseline agent uses **only** `pos` and `mineral_pos`. Everything else is exclusively used by MetaMo.

### Hyperparameters (GridWorld)

| Parameter                    | Default | Smaller → effect              | Larger → effect                        |
|------------------------------|---------|-------------------------------|----------------------------------------|
| `GRID_SIZE`                  | 10      | Easier, faster learning       | Sparser rewards, harder exploration    |
| `MAX_STEPS`                  | 50      | Short episodes, frequent fail | More planning, higher variance         |
| `LAVA_CELL_COUNT`            | 4       | Safer, less conflict          | Dense hazards, survival dominates      |
| `MINERAL_SPAWN_BAND`         | 2       | Fewer risky minerals          | Many minerals near lava                |
| `DANGER_MINERAL_PROBABILITY` | 0.55    | Mostly safe minerals          | Mostly dangerous minerals              |
| `REWARD_MINERAL`             | +100    | Weak motivation               | Greedy, mineral-obsessed behaviour     |
| `REWARD_LAVA`                | −15     | Agent ignores danger          | Strong avoidance, overly cautious      |
| `REWARD_STEP`                | −0.5    | Wandering allowed             | Very direct, shortest-path pressure    |
| `REWARD_BOUNDARY`            | −1      | Almost no wall penalty        | Strict confinement                     |

---

## 4. Baseline Agent — Tabular Q-Learning

`agents/baseline_agent.py`

### What it is

A pure Markov Decision Process (MDP) solver. It learns a mapping from `(agent_pos, mineral_pos)` to the best action using the Bellman update. It has no internal drives, no safety awareness, no arousal — only a static lookup table updated by reward.

### State encoding

```python
def _encode(self, state: dict) -> tuple:
    ar, ac = state["pos"]
    mr, mc = state["mineral_pos"]
    return (ar, ac, mr, mc)
```

The Q-table shape is `(10, 10, 10, 10, 4)` = **40,000 values**.
Each entry `Q[ar, ac, mr, mc, action]` holds the estimated future reward for taking `action` from that configuration.

Example: `Q[2, 3, 8, 9]` → `[-0.5, 2.4, -1.0, 3.1]` → agent chooses action 3 (Right).

### The Bellman update

```
Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]
```

- `td_target = r + γ · max Q(s')`  (or just `r` if terminal)
- `td_error  = td_target − Q(s, a)`
- Update: `Q(s, a) += α · td_error`

### Key current limitation: the state is blind to danger

The baseline encodes `(agent_row, agent_col, mineral_row, mineral_col)` — nothing else. The following information is **completely absent** from its state space:

- `lava_distance`
- `in_lava`
- `lava_cells`
- Energy level
- Arousal or safety signals

So the baseline is fundamentally learning: **"How do I reach the mineral?"** — not **"How do I balance safety and reward?"** When lava spawns between the agent and the mineral, the agent has no way to know it is there until it steps on it and receives a negative reward. It cannot generalise across lava layouts because lava position is not part of its state. This brittle, position-only reasoning is the intended contrast with MetaMo.

### Hyperparameters (Baseline)

| Parameter       | Default | Smaller → effect                        | Larger → effect                          |
|-----------------|---------|-----------------------------------------|------------------------------------------|
| `alpha`         | 0.1     | Stable, learns slowly                   | Learns fast, can become unstable         |
| `gamma`         | 0.95    | Only cares about immediate rewards      | Long-term planning matters more          |
| `epsilon`       | 1.0     | Starts fully explorative (required)     | —                                        |
| `epsilon_min`   | 0.05    | Gets stuck in local optima              | Always partly random, never converges    |
| `epsilon_decay` | 0.995   | Explores longer, slower exploitation    | Becomes greedy quickly                   |

Starting `epsilon = 1.0` is essential. With 40,000 states initialised to zero, the agent must stumble around randomly to collect its first real reward signals before exploitation is meaningful.

---

## 5. MetaMo Agent — Motivated Q-Learning

`agents/metamo_agent.py`

### What it adds over the baseline

The MetaMo agent keeps the same Q-table and Bellman update as the baseline, but wraps the action selection in a motivational scoring layer. Q-values are just one term in a combined score that also includes:

- Motivational alignment with current goals (safety vs growth)
- Risk penalty weighted by the individuation goal `G_IND`
- An exploration bonus from visit counts

### State encoding

Identical to the baseline — `(ar, ac, mr, mc)`. The motivational layer does not touch the Q-table indexing; it influences *which action gets chosen*, not *how Q-values are stored*.

### Additional state

| Attribute         | Type               | Purpose                                        |
|-------------------|--------------------|------------------------------------------------|
| `mot`             | `MotivationalState`| Live goal vector G and modulator vector M      |
| `visit_counts`    | `ndarray`          | Same shape as Q-table, tracks exploration      |
| `_pending_state`  | `MotivationalState`| Proposed next motivational state (applied on update) |
| `log_alpha`       | `list[dict]`       | Per-step decision explanation                  |
| `log_srv`         | `list[bool]`       | Per-step safe-region violation flag            |

### Hyperparameters (MetaMo — on top of Q-learning)

| Parameter                  | Default | Smaller → effect                              | Larger → effect                             |
|----------------------------|---------|-----------------------------------------------|---------------------------------------------|
| `alpha`                    | 0.1     | Same as baseline                              | Same as baseline                            |
| `gamma`                    | 0.95    | Same as baseline                              | Same as baseline                            |
| `epsilon`                  | 1.0     | —                                             | —                                           |
| `epsilon_min`              | 0.10    | Less exploration floor than baseline (5%)     | Higher minimum randomness                   |
| `epsilon_decay`            | 0.995   | Same as baseline                              | Same as baseline                            |
| `motivation_weight`        | 6.0     | MetaMo influence weaker, closer to baseline   | MetaMo fully dominates action selection     |
| `risk_weight`              | 4.0     | Agent ignores risk estimates from candidates  | Agent very risk-averse, slow mineral pickup |
| `exploration_bonus_weight` | 1.25    | Less UCB-style bonus, less systematic coverage | More aggressive unvisited-state seeking    |

---

## 6. Motivational State

`metamo/state.py` and `core/state.py`

The motivational state `MotivationalState(G, M)` is the internal model the MetaMo agent carries at all times. It consists of two vectors.

### Goal vector G (8 goals)

| Index    | Name    | Initial | Likely importance in GridWorld | Meaning                                    |
|----------|---------|---------|--------------------------------|--------------------------------------------|
| `G_IND`  | Individuation | 0.65 | Very High | Self-preservation, safety drive        |
| `G_TRANS`| Transcendence | 0.55 | Very High | Exploration, growth, reward pursuit    |
| `G_HELP` | Helping       | 0.75 | Low       | Cooperative actions (unused here)      |
| `G_CURIO`| Curiosity     | 0.50 | Medium    | Discovering new states                 |
| `G_NOVEL`| Novelty       | 0.45 | Medium    | Seeking unfamiliar situations          |
| `G_SELF` | Self-oriented | 0.30 | Low       | Self-benefit (limited effect here)     |
| `G_ETHIC`| Ethical       | 0.85 | Low       | Avoidance of harmful actions           |
| `G_SOC`  | Social        | 0.20 | Low       | Social objectives (unused here)        |

### Modulator vector M (6 modulators)

Initialised to `0.5` (neutral midpoint). Modulators adjust over time via the appraisal pipeline; for example `M_AROUSAL` rises near lava and `M_THRESHOLD` governs how cautious the agent becomes.

### Safe region S

```
S = { (G, M) : G_IND ≥ THETA_SAFE  AND  ||G|| ≤ G_MAX }
```

The agent is in the safe region when its individuation goal is above the safety floor **and** its total goal magnitude is bounded. Violations of this condition are tracked as SRV (Safe-Region Violations).

---

## 7. MetaMo Core Pipeline

`metamo/core.py` adapts the root MetaMo pseudo-bimonad for the GridWorld.

```
Environment state dict
        ↓
  build_stimulus()          → Stimulus(novelty, conduciveness, risk, effort)
        ↓
  OpenPsi appraisal         → updates modulator M
        ↓
  build_candidates()        → 4 Action objects (one per direction)
        ↓
  build_consensus_states()  → safety_state, growth_state
        ↓
  consensus_candidate_scores() → score per action (from MAGUS decision)
        ↓
  transition_for_action()   → chosen Action, next MotivationalState, target_state
        ↓
  project_to_safe_region()  → clamp G_IND ≥ THETA_SAFE, scale others if needed
        ↓
  blend_states()            → smooth transition (Lipschitz-bounded drift)
        ↓
  New MotivationalState applied on next update()
```

### build_stimulus

Converts the raw env dict into a `Stimulus` object:

| Field           | Formula                                          | Meaning                        |
|-----------------|--------------------------------------------------|--------------------------------|
| `risk`          | `1.0` if in lava, else `clip(0.60 − 0.16 × lava_dist)` | Danger level           |
| `conduciveness` | `clip(1.0 − distance / 18)`                     | How close to the goal          |
| `novelty`       | `clip(0.30 + 0.70 × (1 − distance / 18))`       | How interesting current state  |
| `effort`        | `clip(0.10 + 0.40 × (risk + distance/18) / 2)`  | Cost of acting                 |

### build_candidates

Builds 4 `Action` objects. For each direction the function projects where the agent would land, then computes:

- `risk_estimate`: 1.0 (in lava) → 0.55 (lava_dist ≤ 1) → 0.25 (≤ 2) → 0.05 (safe)
- `goal_correlations`: 8-vector of how well the action serves each goal
- `delta_g`: how the goal vector should shift if this action is taken

### build_consensus_states

Splits the current state into two perspectives:

**Safety perspective** — raises `G_IND` and `G_ETHIC` when risk is high, lowers `G_TRANS`.

**Growth perspective** — raises `G_TRANS`, `G_CURIO`, `G_NOVEL` when opportunity is high.

### consensus_candidate_scores

Scores each action from both perspectives using MAGUS decision logic, then merges:
```
score = (score_safety + score_growth) / 2  −  0.25 × |score_safety − score_growth|
```
The penalty term discourages actions where the two perspectives strongly disagree.

---

## 8. How `select_action` Works in the MetaMo Agent

MetaMo's action selection is **not** "RL picks an action and MetaMo adjusts it." It is a single combined scoring function where Q-learning is one term among several.

```python
# Step 1 — perceive
stimulus   = build_stimulus(state, self.mot)
candidates = build_candidates(state, self.mot)

# Step 2 — score from both motivational perspectives
mot_scores = consensus_candidate_scores(self.mot, stimulus, candidates, state)
# mot_scores: shape (4,), one score per action

# Step 3 — classical RL value
q_values = self.q_table[self._encode(state)]          # shape (4,)

# Step 4 — exploration bonus (UCB-style)
visit_counts    = self.visit_counts[self._encode(state)]
exploration_bonus = exploration_bonus_weight / sqrt(visit_counts + 1)

# Step 5 — combine everything into one score per action
regularized_scores = (
    q_values
    + motivation_weight  * mot_scores
    - risk_weight * G_IND * risk_estimates
    + exploration_bonus
)

# Step 6 — ε-greedy over the combined score
if random() < epsilon:
    action = random_action()
else:
    action = argmax(regularized_scores)

# Step 7 — simulate the motivational transition for the chosen action
action, next_mot_state, stimulus, target_state = transition_for_action(...)
self._pending_state = next_mot_state   # applied in update()
```

### Why this design matters

The raw Q-value alone would choose the shortest path to the mineral regardless of lava. The motivational scores push the combined score toward safer actions when `G_IND` is high and risk is high. The exploration bonus ensures the agent systematically visits unseen state-action pairs rather than relying on random epsilon.

`G_IND` appears twice: as the base goal (raised by the safety perspective) and as the coefficient in the risk penalty term `risk_weight × G_IND × risk_estimates`. This means the stronger the agent's self-preservation drive, the harder lava gets penalised in action selection.

The motivational state is never applied immediately. `_pending_state` is held and only committed inside `update()` after the environment confirms what happened. This keeps the motivational update causally ordered after the environment step.

### alpha dict (decision explanation)

After each `select_action`, the agent logs an `alpha` dict:

| Key                    | Meaning                                              |
|------------------------|------------------------------------------------------|
| `risk`                 | Stimulus risk at current position                    |
| `urgency`              | `1 − energy/100` (how close to dying)                |
| `eu`                   | Expected utility proxy (closeness to mineral)        |
| `individuation`        | Current `G_IND`                                      |
| `transcendence`        | Current `G_TRANS`                                    |
| `target_individuation` | `G_IND` in the proposed next motivational state      |
| `target_transcendence` | `G_TRANS` in the proposed next motivational state    |
| `q_value`              | Raw Q-table value for chosen action                  |
| `motivation_score`     | MetaMo consensus score for chosen action             |
| `exploration_bonus`    | UCB bonus for chosen action                          |
| `combined_score`       | Final regularized score for chosen action            |

This dict is what the dashboard displays as "Appraisal risk", "Consensus Ind/Trans", etc.

---

## 9. Metrics & Collector

`metrics/collector.py`

### EpisodeLog fields

| Field                | Type    | Meaning                                             |
|----------------------|---------|-----------------------------------------------------|
| `minerals_collected` | int     | Minerals the agent collected this episode           |
| `minerals_spawned`   | int     | Minerals the environment spawned this episode       |
| `total_steps`        | int     | Steps taken                                         |
| `total_reward`       | float   | Cumulative reward                                   |
| `lava_steps`         | int     | Steps spent on a lava cell                          |
| `srv_flags`          | list    | Per-step: True = safe-region violation              |
| `unsafe_flags`       | list    | Per-step: True = in lava or danger band             |
| `arousal_log`        | list    | MetaMo only: `M_AROUSAL` per step                  |
| `safety_log`         | list    | MetaMo only: `M_THRESHOLD` per step                |
| `individuation_log`  | list    | MetaMo only: `G_IND` per step                      |
| `transcendence_log`  | list    | MetaMo only: `G_TRANS` per step                    |
| `energy_log`         | list    | Energy level per step                               |

### Metric formulas

**Completion Rate (CR)**
```
CR = minerals_collected / minerals_spawned
```

**Lava Rate**
```
lava_rate = lava_steps / total_steps
```

**SRV Rate** (Safe-Region Violation Rate)
- MetaMo: `True` when `not in_safe_region(mot)` — a motivational check
- Baseline: `True` when `in_lava OR lava_distance ≤ DANGER_DISTANCE` — an environment proxy

These are **not the same measure**. MetaMo's SRV is internal; baseline SRV is external. Comparing them directly is misleading.

**Unsafe Rate**
```
unsafe_rate = steps where (in_lava OR lava_distance ≤ 2) / total_steps
```
This is the same environmental measure for both agents and is the fair comparison for danger exposure.

**Recovery Time (RT)**

For each violation bout starting at step `t0`, recovery is the first time `τ` such that `L = 3` consecutive safe steps occur:

```
RT_i = t_recover − t0
RT   = mean(RT_i) over all bouts
     = total_steps if no recovery ever occurs
```

---

## 10. Simulation Layer

`simulation/main.py` — pygame event loop only. Calls `runner` and `renderer`.

`simulation/runner.py` — all non-visual logic:
- `train_agent()` — training loop
- `new_episode()` — creates two fresh envs with the same seed
- `step_baseline()` / `step_metamo()` — advances one agent by one step, updates EpisodeLog

`simulation/renderer.py` — all pygame drawing:
- `draw_grid()` — lava animation, mineral glow, agent sprite
- `draw_panel()` — dashboard rows, bars, MetaMo vs baseline sections

### Dashboard — what each panel shows

| Row / Bar            | Source                                        |
|----------------------|-----------------------------------------------|
| Episode / Completed  | Episode counter                               |
| Step                 | `env_state["step"]`                           |
| Minerals             | `ep_log.minerals_collected / spawned`         |
| Reward               | Cumulative reward this episode                |
| Lava ep/avg          | Live lava rate / historical mean              |
| Unsafe ep/avg        | Live unsafe rate / historical mean            |
| Env region           | SAFE / DANGER / LAVA from lava_distance       |
| MetaMo SRV ep/avg    | MetaMo only: `not in_safe_region(mot)`        |
| Appraisal risk       | MetaMo only: `alpha["risk"]`                  |
| Individuation bar    | `mot.G[G_IND]` (current)                      |
| Consensus Ind bar    | `alpha["target_individuation"]` (proposed)    |
| Transcendence bar    | `mot.G[G_TRANS]` (current)                    |
| Consensus Trans bar  | `alpha["target_transcendence"]` (proposed)    |
| Safety threshold bar | `mot.M[M_THRESHOLD]`                          |
| Arousal bar          | `mot.M[M_AROUSAL]`                            |
| Energy bar           | Baseline only: `env_state["energy"]`          |

---

## 11. Hyperparameter Reference

### Quick-tune guide

To make MetaMo **more safety-conservative** — raise `G_IND` initial value, raise `risk_weight`, raise `LAVA_CELL_COUNT`.

To make MetaMo **more reward-hungry** — raise `G_TRANS` initial value, raise `motivation_weight`, lower `risk_weight`.

To make training **longer/more stable** — raise `TRAIN_EPISODES`, lower `epsilon_decay` (both agents explore longer).

To make evaluation **easier to observe** — lower `DEFAULT_STEPS_PER_SECOND` in `main.py`, or reduce `EVAL_EPISODES`.

### EVALUATION SUMMARY 


============================================================
  EVALUATION SUMMARY
============================================================

  [ Baseline ]  (50 episodes)
  ──────────────────────────────────────────────────
  Completion rate  :
  Total reward     : 
  Lava rate        : 
  Unsafe-zone rate : 
  Recovery time    : 
  Env SRV (proxy)  :

  [ MetaMo ]  (50 episodes)
  ──────────────────────────────────────────────────
  Completion rate  : 
  Total reward     : 
  Lava rate        : 
  Unsafe-zone rate :  
 

  NOTE: unsafe_rate is the fair cross-agent safety comparison.
        MetaMo 'Motivational SRV' and Baseline 'Env SRV' measure
        different things and should not be compared directly.