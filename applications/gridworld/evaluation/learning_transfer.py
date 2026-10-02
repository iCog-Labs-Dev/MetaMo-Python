"""Checkpointed GridWorld learning-transfer and safety-shielding diagnostic.

Two agents are trained with matched seeds: one using Q-only action selection
and one using MetaMo-guided action selection.  At each requested checkpoint a
deep copy is evaluated with Q only, so evaluation cannot alter continued
training and cannot receive decision-time help from MetaMo.
"""

from __future__ import annotations

import argparse
import copy
import csv
import html
from pathlib import Path

import numpy as np

from applications.gridworld.config import MAX_STEPS
from applications.gridworld.evaluation.metrics import EpisodeLog, MetricsCollector
from applications.gridworld.evaluation.runner import (
    REGIMES,
    REGIME_ORDER,
    _agent_seed_values,
    _evaluate_agent,
    _run_episode,
    _seed_stat,
    _variant_specs,
)


TRAINING_VARIANTS = ("QTrainQEval", "MetaMoTrainQEval")
VARIANT_LABELS = {
    "QTrainQEval": "Q-only training",
    "MetaMoTrainQEval": "MetaMo-guided training",
}
VARIANT_COLORS = {
    "QTrainQEval": "#2563eb",
    "MetaMoTrainQEval": "#16a34a",
}
EVALUATION_METRICS = (
    "total_reward",
    "minerals_collected",
    "completion_rate",
    "lava_rate",
    "unsafe_rate",
    "survival_rate",
    "path_efficiency",
    "q_proposed_lava_rate",
    "executed_lava_entry_rate",
    "q_proposed_risk",
    "executed_risk",
)
DIAGNOSTIC_METRICS = (
    "training_steps",
    "training_lava_rate",
    "training_unsafe_rate",
    "q_proposed_lava_rate",
    "executed_lava_entry_rate",
    "q_proposal_blocked_rate",
    "executed_changed_q_rate",
    "selector_influence_rate",
    "hard_safety_intervention_rate",
    "exploration_rate",
    "q_proposed_risk",
    "executed_risk",
    "q_proposed_visit_count",
    "q_proposed_value",
    "blocked_q_visit_count",
    "blocked_q_value",
    "visited_states",
    "q_table_states",
    "visited_state_actions",
    "state_action_coverage",
    "total_q_updates",
)


def parse_checkpoints(raw: str) -> list[int]:
    """Parse a positive, sorted, unique checkpoint list."""
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or values[0] <= 0:
        raise ValueError("training checkpoints must be positive integers")
    return values


def _flatten(values: list[EpisodeLog], attribute: str) -> list:
    return [item for log in values for item in getattr(log, attribute)]


def _mean(values: list[float | bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def q_coverage(agent) -> dict[str, float]:
    """Summarize executed state-action coverage from Q visit counts."""
    counts = list(agent.visit_counts.values())
    visited_states = sum(float(np.sum(value)) > 0.0 for value in counts)
    visited_pairs = sum(int(np.count_nonzero(value)) for value in counts)
    total_updates = sum(float(np.sum(value)) for value in counts)
    denominator = visited_states * agent.ACTIONS
    return {
        "visited_states": float(visited_states),
        "q_table_states": float(len(agent.q_table)),
        "visited_state_actions": float(visited_pairs),
        "state_action_coverage": (
            float(visited_pairs / denominator) if denominator else 0.0
        ),
        "total_q_updates": total_updates,
    }


def training_diagnostics(
    logs: list[EpisodeLog],
    agent,
) -> dict[str, float]:
    """Return cumulative training diagnostics through the current checkpoint."""
    total_steps = sum(log.total_steps for log in logs)
    lava_steps = sum(log.lava_steps for log in logs)
    unsafe_flags = _flatten(logs, "unsafe_flags")
    blocked_flags = _flatten(logs, "q_proposal_blocked_flags")
    proposed_visits = _flatten(logs, "q_proposed_visit_count_log")
    proposed_values = _flatten(logs, "q_proposed_value_log")
    blocked_visits = [
        value for value, blocked in zip(proposed_visits, blocked_flags) if blocked
    ]
    blocked_values = [
        value for value, blocked in zip(proposed_values, blocked_flags) if blocked
    ]
    result = {
        "training_steps": float(total_steps),
        "training_lava_rate": float(lava_steps / total_steps) if total_steps else 0.0,
        "training_unsafe_rate": _mean(unsafe_flags),
        "q_proposed_lava_rate": _mean(
            _flatten(logs, "q_proposed_lava_flags")
        ),
        "executed_lava_entry_rate": _mean(
            _flatten(logs, "executed_lava_entry_flags")
        ),
        "q_proposal_blocked_rate": _mean(blocked_flags),
        "executed_changed_q_rate": _mean(
            _flatten(logs, "executed_changed_q_flags")
        ),
        "selector_influence_rate": _mean(
            _flatten(logs, "selector_influence_flags")
        ),
        "hard_safety_intervention_rate": _mean(
            _flatten(logs, "hard_safety_intervention_flags")
        ),
        "exploration_rate": _mean(_flatten(logs, "exploration_flags")),
        "q_proposed_risk": _mean(_flatten(logs, "q_proposed_risk_log")),
        "executed_risk": _mean(_flatten(logs, "executed_risk_log")),
        "q_proposed_visit_count": _mean(proposed_visits),
        "q_proposed_value": _mean(proposed_values),
        "blocked_q_visit_count": _mean(blocked_visits),
        "blocked_q_value": _mean(blocked_values),
    }
    result.update(q_coverage(agent))
    return result


def _evaluation_seed_row(
    regime: str,
    danger_probability: float,
    variant_name: str,
    agent_seed: int,
    checkpoint: int,
    logs: list[EpisodeLog],
) -> dict:
    collector = MetricsCollector(
        f"learning-transfer:{regime}:{variant_name}:{agent_seed}:{checkpoint}"
    )
    for log in logs:
        collector.add(log)
    summary = collector.summary()
    row = {
        "regime": regime,
        "danger_mineral_probability": danger_probability,
        "variant": variant_name,
        "training_policy": "q" if variant_name == "QTrainQEval" else "metamo",
        "evaluation_policy": "q",
        "agent_seed": agent_seed,
        "checkpoint": checkpoint,
        "n_eval_episodes": len(logs),
    }
    for metric in EVALUATION_METRICS:
        if metric in summary:
            row[metric] = float(summary[metric]["mean"])
    return row


def aggregate_rows(rows: list[dict], metrics: tuple[str, ...]) -> list[dict]:
    """Aggregate seed rows with seeds as independent experimental units."""
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        key = (str(row["regime"]), str(row["variant"]), int(row["checkpoint"]))
        groups.setdefault(key, []).append(row)

    result: list[dict] = []
    for (regime, variant, checkpoint), group in sorted(
        groups.items(),
        key=lambda item: (
            REGIME_ORDER[item[0][0]],
            TRAINING_VARIANTS.index(item[0][1]),
            item[0][2],
        ),
    ):
        row = {
            "regime": regime,
            "danger_mineral_probability": REGIMES[regime],
            "variant": variant,
            "training_policy": "q" if variant == "QTrainQEval" else "metamo",
            "evaluation_policy": "q",
            "checkpoint": checkpoint,
            "n_agent_seeds": len(group),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in group if metric in item]
            if not values:
                continue
            stat = _seed_stat(values)
            for name, value in stat.items():
                row[f"{metric}_{name}"] = value
        result.append(row)
    return result


def paired_checkpoint_contrasts(
    seed_rows: list[dict],
    metrics: tuple[str, ...] = EVALUATION_METRICS,
    contrast: str = "MetaMoGuidedTraining-QOnlyTraining",
) -> list[dict]:
    """Compute paired MetaMo-guided minus Q-only checkpoint contrasts."""
    indexed = {
        (
            str(row["regime"]),
            str(row["variant"]),
            int(row["checkpoint"]),
            int(row["agent_seed"]),
        ): row
        for row in seed_rows
    }
    results: list[dict] = []
    regimes = sorted({str(row["regime"]) for row in seed_rows}, key=REGIME_ORDER.get)
    checkpoints = sorted({int(row["checkpoint"]) for row in seed_rows})
    seeds = sorted({int(row["agent_seed"]) for row in seed_rows})
    for regime in regimes:
        for checkpoint in checkpoints:
            for metric in metrics:
                differences = []
                for seed in seeds:
                    guided = indexed.get(
                        (regime, "MetaMoTrainQEval", checkpoint, seed)
                    )
                    baseline = indexed.get((regime, "QTrainQEval", checkpoint, seed))
                    if not guided or not baseline:
                        continue
                    if metric not in guided or metric not in baseline:
                        continue
                    differences.append(
                        float(guided[metric]) - float(baseline[metric])
                    )
                if not differences:
                    continue
                stat = _seed_stat(differences)
                results.append({
                    "regime": regime,
                    "checkpoint": checkpoint,
                    "contrast": contrast,
                    "treatment": "MetaMoTrainQEval",
                    "control": "QTrainQEval",
                    "metric": metric,
                    "n_seed_pairs": len(differences),
                    "mean_difference": stat["mean"],
                    "std_difference": stat["std"],
                    "ci95_low": stat["ci95_low"],
                    "ci95_high": stat["ci95_high"],
                })
    return results


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _svg_text(x, y, value, size=12, anchor="middle", weight="400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
        f'fill="#334155">{html.escape(str(value))}</text>'
    )


def _plot_metric_grid(
    rows: list[dict],
    regime: str,
    metrics: tuple[tuple[str, str, bool], ...],
    title: str,
    output: Path,
) -> None:
    selected = [row for row in rows if row["regime"] == regime]
    checkpoints = sorted({int(row["checkpoint"]) for row in selected})
    lookup = {
        (str(row["variant"]), int(row["checkpoint"])): row for row in selected
    }
    width, height = 1040, 720
    panel_w, panel_h = 430, 250
    origins = ((80, 90), (570, 90), (80, 405), (570, 405))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 38, f"{title} — {regime.title()} risk", 22, weight="700"),
    ]
    for (metric, label, force_rate), (x0, y0) in zip(metrics, origins):
        values = []
        for row in selected:
            values.extend([
                float(row.get(f"{metric}_ci95_low", 0.0)),
                float(row.get(f"{metric}_ci95_high", 0.0)),
            ])
        ymin = min(values + [0.0])
        ymax = max(values + [1.0 if force_rate else 0.0])
        if force_rate:
            ymin, ymax = 0.0, min(1.0, max(ymax * 1.08, 0.05))
        elif ymax <= ymin:
            ymax = ymin + 1.0
        else:
            padding = 0.08 * (ymax - ymin)
            ymin -= padding
            ymax += padding

        def sx(checkpoint: int) -> float:
            if len(checkpoints) == 1:
                return x0 + panel_w / 2
            return x0 + (checkpoint - checkpoints[0]) / (
                checkpoints[-1] - checkpoints[0]
            ) * panel_w

        def sy(value: float) -> float:
            return y0 + panel_h - (value - ymin) / (ymax - ymin) * panel_h

        parts.extend([
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + panel_h}" stroke="#64748b"/>',
            f'<line x1="{x0}" y1="{y0 + panel_h}" x2="{x0 + panel_w}" y2="{y0 + panel_h}" stroke="#64748b"/>',
            _svg_text(x0 + panel_w / 2, y0 - 18, label, 15, weight="700"),
        ])
        for tick in range(5):
            value = ymin + (ymax - ymin) * tick / 4
            y = sy(value)
            parts.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_w}" y2="{y:.1f}" stroke="#e2e8f0"/>'
            )
            parts.append(_svg_text(x0 - 9, y + 4, f"{value:.2f}", 10, "end"))
        for checkpoint in checkpoints:
            parts.append(
                _svg_text(sx(checkpoint), y0 + panel_h + 20, checkpoint, 10)
            )
        parts.append(
            _svg_text(x0 + panel_w / 2, y0 + panel_h + 42, "Training episodes", 11)
        )
        for variant in TRAINING_VARIANTS:
            color = VARIANT_COLORS[variant]
            points = []
            for checkpoint in checkpoints:
                row = lookup.get((variant, checkpoint))
                if not row or f"{metric}_mean" not in row:
                    continue
                mean = float(row[f"{metric}_mean"])
                low = float(row[f"{metric}_ci95_low"])
                high = float(row[f"{metric}_ci95_high"])
                x, y = sx(checkpoint), sy(mean)
                points.append(f"{x:.1f},{y:.1f}")
                parts.extend([
                    f'<line x1="{x:.1f}" y1="{sy(low):.1f}" x2="{x:.1f}" y2="{sy(high):.1f}" stroke="{color}" stroke-width="1.5"/>',
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>',
                ])
            if len(points) > 1:
                parts.append(
                    f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>'
                )
    for index, variant in enumerate(TRAINING_VARIANTS):
        x = 330 + index * 260
        color = VARIANT_COLORS[variant]
        parts.append(f'<line x1="{x}" y1="690" x2="{x + 28}" y2="690" stroke="{color}" stroke-width="3"/>')
        parts.append(_svg_text(x + 36, 694, VARIANT_LABELS[variant], 12, "start"))
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def save_plots(
    checkpoint_rows: list[dict],
    diagnostic_rows: list[dict],
    output_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for regime in sorted(
        {str(row["regime"]) for row in checkpoint_rows},
        key=REGIME_ORDER.get,
    ):
        transfer_path = output_dir / f"gridworld_learning_transfer_{regime}.svg"
        _plot_metric_grid(
            checkpoint_rows,
            regime,
            (
                ("total_reward", "Q-only evaluation reward", False),
                ("lava_rate", "Q-only evaluation lava rate", True),
                ("unsafe_rate", "Q-only evaluation unsafe rate", True),
                ("survival_rate", "Q-only evaluation survival", True),
            ),
            "Learning transfer after selector withdrawal",
            transfer_path,
        )
        paths.append(transfer_path)
        diagnostic_path = output_dir / f"gridworld_training_diagnostics_{regime}.svg"
        _plot_metric_grid(
            diagnostic_rows,
            regime,
            (
                ("q_proposed_lava_rate", "Q proposes immediate lava", True),
                ("q_proposal_blocked_rate", "Q proposals blocked", True),
                ("training_lava_rate", "Executed training lava rate", True),
                ("state_action_coverage", "State-action coverage", True),
            ),
            "Training safety-shield diagnostics",
            diagnostic_path,
        )
        paths.append(diagnostic_path)
    return paths


def run(args: argparse.Namespace) -> dict[str, Path]:
    checkpoints = parse_checkpoints(args.train_checkpoints)
    regimes = [part.strip() for part in args.regimes.split(",") if part.strip()]
    unknown = [regime for regime in regimes if regime not in REGIMES]
    if unknown:
        raise ValueError(f"unknown regimes: {unknown}")
    seeds = _agent_seed_values(args.agent_seeds)
    variants = _variant_specs()
    checkpoint_seed_rows: list[dict] = []
    diagnostic_seed_rows: list[dict] = []

    for regime in regimes:
        danger_probability = REGIMES[regime]
        for variant_name in TRAINING_VARIANTS:
            variant = variants[variant_name]
            print(f"\n[{regime}] {VARIANT_LABELS[variant_name]}", flush=True)
            for agent_seed in seeds:
                agent = variant.factory(agent_seed)
                training_logs: list[EpisodeLog] = []
                for episode in range(1, checkpoints[-1] + 1):
                    training_logs.append(
                        _run_episode(
                            agent,
                            variant,
                            env_seed=args.train_seed_start + episode - 1,
                            max_steps=args.max_steps,
                            danger_distance=args.danger_distance,
                            danger_mineral_probability=danger_probability,
                            train=True,
                        )
                    )
                    if episode not in checkpoints:
                        continue
                    probe = copy.deepcopy(agent)
                    probe.epsilon = args.eval_epsilon
                    eval_logs = _evaluate_agent(
                        probe,
                        variant,
                        episodes=args.eval_episodes,
                        seed_start=args.test_seed_start,
                        max_steps=args.max_steps,
                        danger_distance=args.danger_distance,
                        danger_mineral_probability=danger_probability,
                        record_appraisal_counterfactual=False,
                        audit_compositionality=False,
                    )
                    checkpoint_seed_rows.append(
                        _evaluation_seed_row(
                            regime,
                            danger_probability,
                            variant_name,
                            agent_seed,
                            episode,
                            eval_logs,
                        )
                    )
                    diagnostics = training_diagnostics(training_logs, agent)
                    diagnostic_seed_rows.append({
                        "regime": regime,
                        "danger_mineral_probability": danger_probability,
                        "variant": variant_name,
                        "training_policy": variant.policy_for(True),
                        "agent_seed": agent_seed,
                        "checkpoint": episode,
                        **diagnostics,
                    })
                if not args.quiet_seeds:
                    print(
                        f"  seed {agent_seed} complete through episode {checkpoints[-1]}",
                        flush=True,
                    )

    checkpoint_rows = aggregate_rows(checkpoint_seed_rows, EVALUATION_METRICS)
    diagnostic_rows = aggregate_rows(diagnostic_seed_rows, DIAGNOSTIC_METRICS)
    contrast_rows = paired_checkpoint_contrasts(checkpoint_seed_rows)
    diagnostic_contrast_rows = paired_checkpoint_contrasts(
        diagnostic_seed_rows,
        metrics=DIAGNOSTIC_METRICS,
        contrast="MetaMoGuidedTraining-QOnlyTrainingDiagnostic",
    )
    output_dir = Path(args.output_dir)
    paths = {
        "checkpoints": output_dir / "gridworld_learning_checkpoints.csv",
        "checkpoint_seeds": output_dir / "gridworld_learning_checkpoint_seeds.csv",
        "diagnostics": output_dir / "gridworld_learning_diagnostics.csv",
        "diagnostic_summary": output_dir / "gridworld_learning_diagnostic_summary.csv",
        "contrasts": output_dir / "gridworld_learning_contrasts.csv",
        "diagnostic_contrasts": (
            output_dir / "gridworld_learning_diagnostic_contrasts.csv"
        ),
    }
    _write_csv(paths["checkpoints"], checkpoint_rows)
    _write_csv(paths["checkpoint_seeds"], checkpoint_seed_rows)
    _write_csv(paths["diagnostics"], diagnostic_seed_rows)
    _write_csv(paths["diagnostic_summary"], diagnostic_rows)
    _write_csv(paths["contrasts"], contrast_rows)
    _write_csv(paths["diagnostic_contrasts"], diagnostic_contrast_rows)

    plot_paths = save_plots(checkpoint_rows, diagnostic_rows, output_dir / "plots")
    print("\nCheckpoint results:", flush=True)
    for row in checkpoint_rows:
        print(
            "  [{}] {} episode={} reward={:.1f} lava={:.3f} unsafe={:.3f}".format(
                row["regime"],
                VARIANT_LABELS[str(row["variant"])],
                row["checkpoint"],
                row.get("total_reward_mean", 0.0),
                row.get("lava_rate_mean", 0.0),
                row.get("unsafe_rate_mean", 0.0),
            ),
            flush=True,
        )
    for path in paths.values():
        print(f"Wrote {path}", flush=True)
    for path in plot_paths:
        print(f"Saved {path}", flush=True)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-checkpoints", default="10,25,50,100")
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--danger-distance", type=int, default=2)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--test-seed-start", type=int, default=2000)
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument("--agent-seeds", default="0-9")
    parser.add_argument("--regimes", default="high")
    parser.add_argument(
        "--output-dir",
        default="eval_results/gridworld_learning_transfer_high_10seed",
    )
    parser.add_argument("--quiet-seeds", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
