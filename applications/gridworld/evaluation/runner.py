"""
Controlled non-visual GridWorld evaluation runner.

The default variants share their Q representation, reward shaping, exploration
policy, valid-action handling, training schedule, and environment seeds.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from applications.gridworld.agents.baseline import BaselineAgent
from applications.gridworld.agents.metamo import MetaMoAgent
from applications.gridworld.config import MAX_STEPS
from applications.gridworld.environment import GridWorld
from applications.gridworld.evaluation.metrics import EpisodeLog, MetricsCollector
from applications.gridworld.runtime import (
    arousal as mot_arousal,
    in_safe_region as mot_in_safe_region,
    safety_threshold as mot_safety_threshold,
)
from dynamics.stability import (
    boundary_pressure as mot_boundary_pressure,
    is_in_boundary_band as mot_in_boundary_band,
)


REGIMES = {
    "low": 0.20,
    "medium": 0.55,
    "high": 0.80,
}
REGIME_ORDER = {name: idx for idx, name in enumerate(REGIMES)}
VARIANT_ORDER = {
    "BaselineCompactQ": 0,
    "MetaMoTaskSelector": 1,
    "MetaMoSafetySelector": 2,
    "MetaMoComposedSelector": 3,
    "MetaMo": 4,
    "QTrainQEval": 5,
    "MetaMoTrainMetaMoEval": 6,
    "MetaMoTrainQEval": 7,
    "QTrainMetaMoEval": 8,
    "BaselineCompactQRaw": 9,
    "BaselineSafetyPriorQ": 10,
}

FAIR_LEARNING_CONFIG = {
    "q_encoder": "compact_hazard",
    "reward_mode": "shaped",
    "safe_exploration_probability": 0.7,
    "mask_lava_on_exploit": False,
    "allow_boundary_actions": False,
}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    kind: str
    factory: Callable[[int], object]
    train_policy: str | None = None
    eval_policy: str | None = None
    training_group: str | None = None

    def policy_for(self, train: bool) -> str:
        """Return the action policy enabled in the requested phase."""
        configured = self.train_policy if train else self.eval_policy
        if configured is not None:
            return configured
        return "metamo" if self.kind == "metamo" else "q"


def _agent_seed_values(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return values or [0]


def _variant_specs() -> dict[str, VariantSpec]:
    variants = {
        "BaselineCompactQ": VariantSpec(
            name="BaselineCompactQ",
            kind="baseline",
            factory=lambda seed: BaselineAgent(
                seed=seed,
                **FAIR_LEARNING_CONFIG,
            ),
        ),
        "BaselineCompactQRaw": VariantSpec(
            name="BaselineCompactQRaw",
            kind="baseline",
            factory=lambda seed: BaselineAgent(
                seed=seed,
                q_encoder="compact_hazard",
                reward_mode="raw",
                mask_lava_on_exploit=False,
            ),
        ),
        "BaselineSafetyPriorQ": VariantSpec(
            name="BaselineSafetyPriorQ",
            kind="baseline",
            factory=lambda seed: BaselineAgent(
                seed=seed,
                q_encoder="compact_hazard",
                reward_mode="raw",
                safe_exploration_probability=1.0,
                mask_lava_on_exploit=True,
            ),
        ),
        "MetaMo": VariantSpec(
            name="MetaMo",
            kind="metamo",
            factory=lambda seed: MetaMoAgent(
                seed=seed,
                selector_mode="composed",
                hard_safety=True,
                exploration_bonus_weight=0.0,
                **FAIR_LEARNING_CONFIG,
            ),
        ),
        "MetaMoTaskSelector": VariantSpec(
            name="MetaMoTaskSelector",
            kind="metamo",
            factory=lambda seed: MetaMoAgent(
                seed=seed,
                selector_mode="task",
                hard_safety=False,
                risk_weight=0.0,
                exploration_bonus_weight=0.0,
                **FAIR_LEARNING_CONFIG,
            ),
        ),
        "MetaMoSafetySelector": VariantSpec(
            name="MetaMoSafetySelector",
            kind="metamo",
            factory=lambda seed: MetaMoAgent(
                seed=seed,
                selector_mode="safety",
                hard_safety=False,
                risk_weight=0.0,
                exploration_bonus_weight=0.0,
                **FAIR_LEARNING_CONFIG,
            ),
        ),
        "MetaMoComposedSelector": VariantSpec(
            name="MetaMoComposedSelector",
            kind="metamo",
            factory=lambda seed: MetaMoAgent(
                seed=seed,
                selector_mode="composed",
                hard_safety=False,
                risk_weight=0.0,
                exploration_bonus_weight=0.0,
                **FAIR_LEARNING_CONFIG,
            ),
        ),
    }
    withdrawal_factory = lambda seed: MetaMoAgent(
        seed=seed,
        selector_mode="composed",
        hard_safety=True,
        exploration_bonus_weight=0.0,
        **FAIR_LEARNING_CONFIG,
    )
    variants.update({
        "QTrainQEval": VariantSpec(
            name="QTrainQEval",
            kind="metamo",
            factory=withdrawal_factory,
            train_policy="q",
            eval_policy="q",
            training_group="selector_withdrawal_q",
        ),
        "MetaMoTrainMetaMoEval": VariantSpec(
            name="MetaMoTrainMetaMoEval",
            kind="metamo",
            factory=withdrawal_factory,
            train_policy="metamo",
            eval_policy="metamo",
            training_group="selector_withdrawal_metamo",
        ),
        "MetaMoTrainQEval": VariantSpec(
            name="MetaMoTrainQEval",
            kind="metamo",
            factory=withdrawal_factory,
            train_policy="metamo",
            eval_policy="q",
            training_group="selector_withdrawal_metamo",
        ),
        "QTrainMetaMoEval": VariantSpec(
            name="QTrainMetaMoEval",
            kind="metamo",
            factory=withdrawal_factory,
            train_policy="q",
            eval_policy="metamo",
            training_group="selector_withdrawal_q",
        ),
    })
    return variants


def _in_environment_unsafe_zone(env_state: dict, danger_distance: int) -> bool:
    return bool(env_state["in_lava"] or env_state["lava_distance"] <= danger_distance)


def _make_env(
    seed: int,
    max_steps: int,
    danger_mineral_probability: float,
) -> GridWorld:
    return GridWorld(
        seed=seed,
        max_steps=max_steps,
        danger_mineral_probability=danger_mineral_probability,
    )


def _record_common_step(
    log: EpisodeLog,
    before: dict,
    after: dict,
    env: GridWorld,
    total_reward: float,
    lava_steps: int,
    danger_distance: int,
) -> None:
    log.total_steps = env.step_count
    log.total_reward = total_reward
    log.lava_steps = lava_steps
    log.movement_steps += int(after["pos"] != before["pos"])
    log.minerals_spawned = env.minerals_spawned
    log.energy_log.append(after["energy"])
    log.survived = after["energy"] > 0
    unsafe = _in_environment_unsafe_zone(after, danger_distance)
    log.unsafe_flags.append(unsafe)


def _run_episode(
    agent,
    variant: VariantSpec,
    env_seed: int,
    max_steps: int,
    danger_distance: int,
    danger_mineral_probability: float,
    train: bool,
    audit_compositionality: bool = False,
) -> EpisodeLog:
    env = _make_env(env_seed, max_steps, danger_mineral_probability)
    state = env.reset()
    agent.reset_episode()
    log = EpisodeLog()
    total_reward = 0.0
    lava_steps = 0

    policy = variant.policy_for(train)
    if policy not in {"q", "metamo"}:
        raise ValueError(f"unknown {('training' if train else 'evaluation')} policy: {policy}")
    if policy == "metamo" and variant.kind != "metamo":
        raise ValueError("MetaMo policy requires a MetaMo agent")

    for _ in range(max_steps):
        if policy == "metamo":
            action, alpha = agent.select_action(
                state,
                record_compositionality=(not train and audit_compositionality),
            )
        elif variant.kind == "metamo":
            action, alpha = agent.select_q_action(
                state,
                return_diagnostics=True,
            )
        else:
            action = agent.select_action(state)
            alpha = {}

        next_state, reward, done, info = env.step(action)

        if train:
            if variant.kind == "metamo":
                agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    info.get("event"),
                    alpha,
                )
            else:
                agent.update(state, action, reward, next_state, done)
        elif policy == "metamo":
            # Evaluation freezes Q while preserving MetaMo's state transition.
            agent.commit_motivational_transition()

        total_reward += reward
        if next_state["in_lava"]:
            lava_steps += 1

        if info.get("event") == "mineral":
            log.minerals_collected += 1

        _record_common_step(
            log,
            state,
            next_state,
            env,
            total_reward,
            lava_steps,
            danger_distance,
        )

        if alpha and "q_proposed_immediate_lava" in alpha:
            log.q_proposed_lava_flags.append(
                bool(alpha["q_proposed_immediate_lava"])
            )
            log.executed_lava_entry_flags.append(
                bool(alpha["executed_immediate_lava"])
            )
            log.q_proposal_blocked_flags.append(
                bool(alpha["q_proposal_blocked"])
            )
            log.executed_changed_q_flags.append(
                bool(alpha["executed_changed_q_action"])
            )
            log.q_proposed_risk_log.append(
                float(alpha["q_proposed_hazard_risk"])
            )
            log.executed_risk_log.append(
                float(alpha["executed_hazard_risk"])
            )
            log.q_proposed_visit_count_log.append(
                float(alpha["q_proposed_visit_count"])
            )
            log.q_proposed_value_log.append(float(alpha["q_proposed_value"]))

        if policy == "metamo":
            log.mot_srv_flags.append(not mot_in_safe_region(agent.mot))
            log.mot_boundary_flags.append(mot_in_boundary_band(agent.mot))
            log.mot_pressure_log.append(mot_boundary_pressure(agent.mot))
            log.arousal_log.append(mot_arousal(agent.mot))
            log.safety_log.append(mot_safety_threshold(agent.mot))
            log.individuation_log.append(agent.mot.goal("individuation"))
            log.transcendence_log.append(agent.mot.goal("transcendence"))
            log.exploration_flags.append(bool(alpha["exploratory"]))
            log.risk_penalty_log.append(float(alpha["risk_penalty"]))
            log.selector_influence_flags.append(
                bool(alpha["metamo_changed_q_action"])
            )
            log.hard_safety_intervention_flags.append(
                bool(alpha["hard_safety_intervened"])
            )
            log.q_regret_log.append(float(alpha["q_regret"]))
            log.task_score_log.append(float(alpha["task_score"]))
            log.safety_score_log.append(float(alpha["safety_score"]))
            log.task_safety_disagreement_log.append(
                float(alpha["task_safety_disagreement"])
            )
            log.q_shortlist_size_log.append(float(alpha["q_shortlist_size"]))
            if alpha["compositionality_evaluated"]:
                log.compositionality_error_log.append(
                    float(alpha["compositionality_error"])
                )
                log.compositionality_holds_flags.append(
                    bool(alpha["compositionality_holds"])
                )
                log.compositionality_goal_error_log.append(
                    float(alpha["compositionality_max_goal_error"])
                )
                log.compositionality_modulator_error_log.append(
                    float(alpha["compositionality_max_modulator_error"])
                )
                log.compositionality_action_holds_flags.append(
                    bool(alpha["compositionality_action_holds"])
                )
                log.compositionality_goal_coordinate_log.append(
                    str(alpha["compositionality_max_goal_coordinate"])
                )
                log.compositionality_modulator_coordinate_log.append(
                    str(alpha["compositionality_max_modulator_coordinate"])
                )
        else:
            log.env_srv_flags.append(
                _in_environment_unsafe_zone(next_state, danger_distance)
            )
            if alpha and "exploratory" in alpha:
                log.exploration_flags.append(bool(alpha["exploratory"]))

        state = next_state
        if done:
            break

    log.total_reward = total_reward
    log.lava_steps = lava_steps
    log.total_steps = env.step_count
    log.minerals_spawned = env.minerals_spawned
    log.survived = state["energy"] > 0

    if train:
        agent.decay_epsilon()

    return log


def _train_agent(
    agent,
    variant: VariantSpec,
    episodes: int,
    seed_start: int,
    max_steps: int,
    danger_distance: int,
    danger_mineral_probability: float,
    evaluation_epsilon: float = 0.0,
) -> None:
    for ep in range(episodes):
        _run_episode(
            agent,
            variant,
            env_seed=seed_start + ep,
            max_steps=max_steps,
            danger_distance=danger_distance,
            danger_mineral_probability=danger_mineral_probability,
            train=True,
        )
    agent.epsilon = evaluation_epsilon


def _evaluate_agent(
    agent,
    variant: VariantSpec,
    episodes: int,
    seed_start: int,
    max_steps: int,
    danger_distance: int,
    danger_mineral_probability: float,
    audit_compositionality: bool = True,
) -> list[EpisodeLog]:
    logs: list[EpisodeLog] = []
    for ep in range(episodes):
        logs.append(
            _run_episode(
                agent,
                variant,
                env_seed=seed_start + ep,
                max_steps=max_steps,
                danger_distance=danger_distance,
                danger_mineral_probability=danger_mineral_probability,
                train=False,
                audit_compositionality=audit_compositionality,
            )
        )
    return logs


def _flatten_summary_row(
    regime: str,
    danger_probability: float,
    variant: str,
    summary: dict,
) -> dict:
    row = {
        "regime": regime,
        "danger_mineral_probability": danger_probability,
        "variant": variant,
        "n_episodes": summary.get("n_episodes", 0),
        "n_agent_seeds": summary.get("n_agent_seeds", 0),
    }
    for metric, value in summary.items():
        if not isinstance(value, dict):
            continue
        for stat_name, stat_value in value.items():
            row[f"{metric}_{stat_name}"] = stat_value
    return row


def _flatten_seed_row(
    regime: str,
    danger_probability: float,
    variant: str,
    agent_seed: int,
    summary: dict,
) -> dict:
    row = {
        "regime": regime,
        "danger_mineral_probability": danger_probability,
        "variant": variant,
        "agent_seed": agent_seed,
        "n_episodes": summary.get("n_episodes", 0),
    }
    for metric, value in summary.items():
        if isinstance(value, dict) and "mean" in value:
            row[metric] = value["mean"]
    return row


_T_CRITICAL_95 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
    2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101,
    2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052,
    2.048, 2.045, 2.042,
)


def _seed_stat(values: list[float]) -> dict:
    """Mean and 95% t interval across independently trained agent seeds."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    df = len(values) - 1
    critical = _T_CRITICAL_95[df - 1] if df <= len(_T_CRITICAL_95) else 1.96
    half_width = critical * std / math.sqrt(len(values))
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def _aggregate_seed_summaries(seed_summaries: list[dict]) -> dict:
    """Aggregate episode means using trained agent seeds as experimental units."""
    result = {
        "n_episodes": sum(int(summary.get("n_episodes", 0)) for summary in seed_summaries),
        "n_agent_seeds": len(seed_summaries),
    }
    metric_names = sorted({
        metric
        for summary in seed_summaries
        for metric, value in summary.items()
        if isinstance(value, dict) and "mean" in value
    })
    for metric in metric_names:
        values = [
            float(summary[metric]["mean"])
            for summary in seed_summaries
            if metric in summary
        ]
        result[metric] = _seed_stat(values)
    return result


def _episode_row(
    regime: str,
    danger_probability: float,
    variant: str,
    agent_seed: int,
    episode_idx: int,
    log: EpisodeLog,
) -> dict:
    return {
        "regime": regime,
        "danger_mineral_probability": danger_probability,
        "variant": variant,
        "agent_seed": agent_seed,
        "episode": episode_idx,
        "completion_rate": log.completion_rate(),
        "total_reward": log.total_reward,
        "lava_rate": log.lava_rate(),
        "unsafe_rate": log.unsafe_rate(),
        "recovery_time": log.recovery_time(),
        "final_energy": log.final_energy(),
        "survival_rate": log.survival_rate(),
        "path_efficiency": log.path_efficiency(),
        "mot_srv_rate": log.mot_srv_rate(),
        "env_srv_rate": log.env_srv_rate(),
        "mot_boundary_rate": log.mot_boundary_rate(),
        "mot_pressure": log.mean_mot_pressure(),
        "exploration_rate": log.exploration_rate(),
        "risk_penalty": log.mean_risk_penalty(),
        "selector_influence_rate": log.selector_influence_rate(),
        "hard_safety_intervention_rate": log.hard_safety_intervention_rate(),
        "q_regret": log.mean_q_regret(),
        "task_score": log.mean_task_score(),
        "safety_score": log.mean_safety_score(),
        "task_safety_disagreement": log.mean_task_safety_disagreement(),
        "q_shortlist_size": log.mean_q_shortlist_size(),
        "q_proposed_lava_rate": log.q_proposed_lava_rate(),
        "executed_lava_entry_rate": log.executed_lava_entry_rate(),
        "q_proposal_blocked_rate": log.q_proposal_blocked_rate(),
        "executed_changed_q_rate": log.executed_changed_q_rate(),
        "q_proposed_risk": log.mean_q_proposed_risk(),
        "executed_risk": log.mean_executed_risk(),
        "q_proposed_visit_count": log.mean_q_proposed_visit_count(),
        "q_proposed_value": log.mean_q_proposed_value(),
        "compositionality_error": log.mean_compositionality_error(),
        "compositionality_median_error": log.median_compositionality_error(),
        "compositionality_p95_error": log.p95_compositionality_error(),
        "compositionality_max_error": log.max_compositionality_error(),
        "compositionality_holds_rate": log.compositionality_holds_rate(),
        "compositionality_goal_error": log.mean_compositionality_goal_error(),
        "compositionality_modulator_error": (
            log.mean_compositionality_modulator_error()
        ),
        "compositionality_action_holds_rate": (
            log.compositionality_action_holds_rate()
        ),
        "compositionality_dominant_goal": (
            log.dominant_compositionality_goal_coordinate()
        ),
        "compositionality_dominant_modulator": (
            log.dominant_compositionality_modulator_coordinate()
        ),
        "minerals_collected": log.minerals_collected,
        "minerals_spawned": log.minerals_spawned,
        "total_steps": log.total_steps,
        "movement_steps": log.movement_steps,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    rows = sorted(rows, key=_row_sort_key)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _row_sort_key(row: dict) -> tuple:
    return (
        REGIME_ORDER.get(str(row.get("regime", "")), 99),
        VARIANT_ORDER.get(str(row.get("variant", "")), 99),
        _safe_int(row.get("agent_seed")),
        _safe_int(row.get("episode")),
    )


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _merge_rows(
    existing_rows: list[dict],
    new_rows: list[dict],
    key_fields: tuple[str, ...],
) -> list[dict]:
    merged: dict[tuple[str, ...], dict] = {}
    for row in existing_rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        merged[key] = row
    for row in new_rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        merged[key] = row
    return list(merged.values())


def run(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    variants = _variant_specs()
    selected_variants = [
        name.strip()
        for name in args.variants.split(",")
        if name.strip()
    ]
    selected_regimes = [
        name.strip()
        for name in args.regimes.split(",")
        if name.strip()
    ]
    agent_seeds = _agent_seed_values(args.agent_seeds)

    unknown_variants = [name for name in selected_variants if name not in variants]
    unknown_regimes = [name for name in selected_regimes if name not in REGIMES]
    if unknown_variants:
        raise ValueError(f"unknown variants: {unknown_variants}")
    if unknown_regimes:
        raise ValueError(f"unknown regimes: {unknown_regimes}")

    output_dir = Path(args.output_dir)
    summary_path = output_dir / "gridworld_fair_summary.csv"
    seed_summary_path = output_dir / "gridworld_fair_seed_summary.csv"
    episodes_path = output_dir / "gridworld_fair_episodes.csv"

    if args.append_existing:
        summary_rows = _read_csv(summary_path)
        seed_summary_rows = _read_csv(seed_summary_path)
        episode_rows = _read_csv(episodes_path)
    else:
        summary_rows: list[dict] = []
        seed_summary_rows: list[dict] = []
        episode_rows: list[dict] = []
    trained_agent_cache: dict[tuple[str, str, int], object] = {}

    for regime in selected_regimes:
        danger_probability = REGIMES[regime]
        for variant_name in selected_variants:
            variant = variants[variant_name]
            collector = MetricsCollector(f"{variant_name}:{regime}")
            variant_episode_rows: list[dict] = []
            variant_seed_rows: list[dict] = []
            variant_seed_summaries: list[dict] = []
            print(f"\n[{regime}] {variant_name}", flush=True)
            print(
                f"  train_policy={variant.policy_for(True)}  "
                f"eval_policy={variant.policy_for(False)}",
                flush=True,
            )

            for agent_seed in agent_seeds:
                cache_key = (
                    regime,
                    variant.training_group or "",
                    agent_seed,
                )
                if (
                    variant.training_group is not None
                    and cache_key in trained_agent_cache
                ):
                    agent = copy.deepcopy(trained_agent_cache[cache_key])
                else:
                    agent = variant.factory(agent_seed)
                    _train_agent(
                        agent,
                        variant,
                        episodes=args.train_episodes,
                        seed_start=args.train_seed_start,
                        max_steps=args.max_steps,
                        danger_distance=args.danger_distance,
                        danger_mineral_probability=danger_probability,
                        evaluation_epsilon=args.eval_epsilon,
                    )
                    if variant.training_group is not None:
                        agent.reset_episode()
                        trained_agent_cache[cache_key] = copy.deepcopy(agent)
                logs = _evaluate_agent(
                    agent,
                    variant,
                    episodes=args.eval_episodes,
                    seed_start=args.test_seed_start,
                    max_steps=args.max_steps,
                    danger_distance=args.danger_distance,
                    danger_mineral_probability=danger_probability,
                    audit_compositionality=(not args.no_compositionality_audit),
                )
                seed_collector = MetricsCollector(
                    f"{variant_name}:{regime}:seed={agent_seed}"
                )
                for idx, log in enumerate(logs, start=1):
                    collector.add(log)
                    seed_collector.add(log)
                    variant_episode_rows.append(
                        _episode_row(
                            regime,
                            danger_probability,
                            variant_name,
                            agent_seed,
                            idx,
                            log,
                        )
                    )
                seed_summary = seed_collector.summary()
                variant_seed_summaries.append(seed_summary)
                variant_seed_rows.append(
                    _flatten_seed_row(
                        regime,
                        danger_probability,
                        variant_name,
                        agent_seed,
                        seed_summary,
                    )
                )
                if not args.quiet_seeds:
                    print(
                        f"    seed {agent_seed} complete "
                        f"({args.train_episodes} train, {args.eval_episodes} eval)",
                        flush=True,
                    )

            summary = _aggregate_seed_summaries(variant_seed_summaries)
            summary_row = _flatten_summary_row(
                regime,
                danger_probability,
                variant_name,
                summary,
            )
            summary_row["train_policy"] = variant.policy_for(True)
            summary_row["eval_policy"] = variant.policy_for(False)
            for row in variant_seed_rows:
                row["train_policy"] = variant.policy_for(True)
                row["eval_policy"] = variant.policy_for(False)
            for row in variant_episode_rows:
                row["train_policy"] = variant.policy_for(True)
                row["eval_policy"] = variant.policy_for(False)
            summary_rows = _merge_rows(
                summary_rows,
                [summary_row],
                ("regime", "variant"),
            )
            episode_rows = _merge_rows(
                episode_rows,
                variant_episode_rows,
                ("regime", "variant", "agent_seed", "episode"),
            )
            seed_summary_rows = _merge_rows(
                seed_summary_rows,
                variant_seed_rows,
                ("regime", "variant", "agent_seed"),
            )
            print(
                "  reward={:.1f} [{:.1f}, {:.1f}]  completion={:.3f}  "
                "lava={:.3f}  unsafe={:.3f}  survival={:.3f}".format(
                    summary["total_reward"]["mean"],
                    summary["total_reward"]["ci95_low"],
                    summary["total_reward"]["ci95_high"],
                    summary["completion_rate"]["mean"],
                    summary["lava_rate"]["mean"],
                    summary["unsafe_rate"]["mean"],
                    summary["survival_rate"]["mean"],
                ),
                flush=True,
            )
            if "selector_influence_rate" in summary:
                print(
                    "  selector_influence={:.3f}  q_regret={:.3f}  "
                    "task_safety_disagreement={:.3f}  comp_holds={:.3f}  "
                    "comp_p95={:.4f}  comp_action_holds={:.3f}".format(
                        summary["selector_influence_rate"]["mean"],
                        summary["q_regret"]["mean"],
                        summary["task_safety_disagreement"]["mean"],
                        summary.get("compositionality_holds_rate", {"mean": 0.0})["mean"],
                        summary.get("compositionality_p95_error", {"mean": 0.0})["mean"],
                        summary.get(
                            "compositionality_action_holds_rate",
                            {"mean": 0.0},
                        )["mean"],
                    ),
                    flush=True,
                )
            _write_csv(summary_path, summary_rows)
            _write_csv(seed_summary_path, seed_summary_rows)
            _write_csv(episodes_path, episode_rows)
            print(f"  checkpoint wrote {summary_path}", flush=True)

    _write_csv(summary_path, summary_rows)
    _write_csv(seed_summary_path, seed_summary_rows)
    _write_csv(episodes_path, episode_rows)
    print(f"\nWrote {summary_path}", flush=True)
    print(f"Wrote {seed_summary_path}", flush=True)
    print(f"Wrote {episodes_path}", flush=True)
    return summary_rows, episode_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--danger-distance", type=int, default=2)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--test-seed-start", type=int, default=2000)
    parser.add_argument(
        "--eval-epsilon",
        type=float,
        default=0.0,
        help="evaluation exploration rate; 0 freezes the primary greedy policy",
    )
    parser.add_argument(
        "--no-compositionality-audit",
        action="store_true",
        help="disable the sampled Principle 3 law audit",
    )
    parser.add_argument(
        "--agent-seeds",
        default="0",
        help="comma-separated seeds or ranges, e.g. 0,1,2 or 0-9",
    )
    parser.add_argument(
        "--regimes",
        default="medium",
        help="comma-separated subset of low,medium,high",
    )
    parser.add_argument(
        "--variants",
        default=(
            "BaselineCompactQ,MetaMoTaskSelector,MetaMoSafetySelector,"
            "MetaMoComposedSelector,MetaMo"
        ),
    )
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="merge with existing CSV rows in the output directory",
    )
    parser.add_argument(
        "--quiet-seeds",
        action="store_true",
        help="suppress per-agent-seed progress messages",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
