"""
training loop and episode-lifecycle helpers

Exports:- 

train_agent(agent, label, episodes, seed_offset)
new_episode(episode, baseline, metamo, max_steps)
clear_episode_logs()
in_environment_unsafe_zone(env_state)
environment_region(env_state)
mean_from_summary(summary, key)
lava_cells_from_state(env_state)
"""

import sys, os

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
for path in (REPO_ROOT, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from environment.gridworld import GridWorld, LAVA_CELLS, MAX_STEPS, MINERAL_SPAWN_BAND
from agents.metamo_agent   import MetaMoAgent
from metrics.collector     import EpisodeLog
from metamo.core           import in_safe_region as mot_in_safe_region
from core.config           import G_IND, G_TRANS

# Constants  

TRAIN_EPISODES = 50
EVAL_EPISODES  = 50
MAX_STEPS_EP   = MAX_STEPS
DANGER_DISTANCE = MINERAL_SPAWN_BAND

# Colours referenced by renderer (re-exported for convenience)
RED   = (220,  60,  60)
GREEN = ( 80, 200, 120)
AMBER = (255, 180,  60)


# Pure helpers  

def lava_cells_from_state(env_state: dict) -> tuple:
    return tuple(env_state.get("lava_cells", LAVA_CELLS))


def in_environment_unsafe_zone(env_state: dict) -> bool:
    return bool(env_state["in_lava"] or env_state["lava_distance"] <= DANGER_DISTANCE)


def environment_region(env_state: dict) -> tuple:
    """Return (label, colour) for the agent's current environment zone."""
    if env_state["in_lava"]:
        return "LAVA",   RED
    if env_state["lava_distance"] <= DANGER_DISTANCE:
        return "DANGER", AMBER
    return "SAFE", GREEN


def mean_from_summary(summary: dict, key: str) -> float:
    if not summary:
        return 0.0
    return float(summary.get(key, {}).get("mean", 0.0))


# Training  

def train_agent(agent, label: str, episodes: int, seed_offset: int = 0):
    print(f"Training {label} for {episodes} episodes ...", end="", flush=True)

    for ep in range(episodes):
        env   = GridWorld(seed=ep + seed_offset, max_steps=MAX_STEPS_EP)
        state = env.reset()
        agent.reset_episode()
        done  = False

        for _ in range(MAX_STEPS_EP):
            if isinstance(agent, MetaMoAgent):
                action, alpha = agent.select_action(state)
            else:
                action = agent.select_action(state)
                alpha  = None

            next_state, reward, done, info = env.step(action)

            if isinstance(agent, MetaMoAgent):
                agent.update(state, action, reward, next_state, done,
                             info.get("event"), alpha)
            else:
                agent.update(state, action, reward, next_state, done)

            state = next_state
            if done:
                break

        agent.decay_epsilon()
        if (ep + 1) % 50 == 0:
            print(".", end="", flush=True)

    print(" done.")
    # Lower epsilon for evaluation
    agent.epsilon = 0.12 if isinstance(agent, MetaMoAgent) else 0.05


#   Episode lifecycle  

def new_episode(ep_num: int, baseline, metamo):
    """Create two fresh envs with the same seed and reset both agents."""
    seed    = ep_num * 7
    env_bl  = GridWorld(seed=seed, max_steps=MAX_STEPS_EP)
    env_mm  = GridWorld(seed=seed, max_steps=MAX_STEPS_EP)
    baseline.reset_episode()
    metamo.reset_episode()
    s_bl = env_bl.reset()
    s_mm = env_mm.reset()
    return env_bl, env_mm, s_bl, s_mm


def clear_episode_logs():
    """Return a zeroed episode-state tuple."""
    return False, False, 0.0, 0.0, 0, 0, EpisodeLog(), EpisodeLog()


#   Per-step update helpers  

def step_baseline(baseline, env_bl, s_bl, ep_log_bl, reward_bl, lava_bl, clank_fn=None):
    """Advance baseline by one step. Returns (s_bl, reward_bl, lava_bl, done)."""
    from metamo.core import arousal as mot_arousal, safety_threshold as mot_safety_threshold

    a_bl = baseline.select_action(s_bl)
    ns_bl, r_bl, done_bl, info_bl = env_bl.step(a_bl)
    baseline.update(s_bl, a_bl, r_bl, ns_bl, done_bl)

    reward_bl += r_bl
    if ns_bl["in_lava"]:
        lava_bl += 1
        if clank_fn:
            clank_fn()

    if info_bl.get("event") == "mineral":
        ep_log_bl.minerals_collected += 1

    ep_log_bl.total_steps      = env_bl.step_count
    ep_log_bl.total_reward     = reward_bl
    ep_log_bl.lava_steps       = lava_bl
    ep_log_bl.minerals_spawned = env_bl.minerals_spawned
    ep_log_bl.energy_log.append(ns_bl["energy"])
    ep_log_bl.survived         = ns_bl["energy"] > 0
    bl_unsafe = in_environment_unsafe_zone(ns_bl)
    ep_log_bl.unsafe_flags.append(bl_unsafe)
    ep_log_bl.srv_flags.append(bl_unsafe)

    return ns_bl, reward_bl, lava_bl, done_bl


def step_metamo(metamo, env_mm, s_mm, ep_log_mm, reward_mm, lava_mm, clank_fn=None):
    """Advance MetaMo by one step. Returns (s_mm, reward_mm, lava_mm, done, alpha_mm)."""
    from metamo.core import (
        arousal          as mot_arousal,
        in_safe_region   as mot_in_safe_region,
        safety_threshold as mot_safety_threshold,
    )

    a_mm, alpha_mm = metamo.select_action(s_mm)
    ns_mm, r_mm, done_mm, info_mm = env_mm.step(a_mm)
    metamo.update(s_mm, a_mm, r_mm, ns_mm, done_mm, info_mm.get("event"), alpha_mm)

    reward_mm += r_mm
    if ns_mm["in_lava"]:
        lava_mm += 1
        if clank_fn:
            clank_fn()

    if info_mm.get("event") == "mineral":
        ep_log_mm.minerals_collected += 1

    ep_log_mm.total_steps      = env_mm.step_count
    ep_log_mm.total_reward     = reward_mm
    ep_log_mm.lava_steps       = lava_mm
    ep_log_mm.minerals_spawned = env_mm.minerals_spawned
    ep_log_mm.energy_log.append(ns_mm["energy"])
    ep_log_mm.survived         = ns_mm["energy"] > 0
    ep_log_mm.unsafe_flags.append(in_environment_unsafe_zone(ns_mm))
    ep_log_mm.srv_flags.append(not mot_in_safe_region(metamo.mot))
    ep_log_mm.arousal_log.append(mot_arousal(metamo.mot))
    ep_log_mm.safety_log.append(mot_safety_threshold(metamo.mot))
    ep_log_mm.individuation_log.append(metamo.mot.G[G_IND])
    ep_log_mm.transcendence_log.append(metamo.mot.G[G_TRANS])

    return ns_mm, reward_mm, lava_mm, done_mm, alpha_mm
