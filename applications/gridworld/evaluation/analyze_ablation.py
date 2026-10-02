"""Compute paired, seed-level contrasts for GridWorld selector experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from applications.gridworld.evaluation.runner import REGIME_ORDER, _seed_stat


CONTRASTS = (
    (
        "MetaMoTrainQEval",
        "QTrainQEval",
        "GuidedTraining-QEval",
    ),
    (
        "QTrainMetaMoEval",
        "QTrainQEval",
        "MetaMoAtEvaluation",
    ),
    (
        "MetaMoTrainMetaMoEval",
        "MetaMoTrainQEval",
        "KeepMetaMoAfterGuidedTraining",
    ),
    (
        "MetaMoTrainMetaMoEval",
        "QTrainMetaMoEval",
        "GuidedTraining-MetaMoEval",
    ),
    (
        "MetaMoTrainMetaMoEval",
        "QTrainQEval",
        "FullMetaMo-TotalEffect",
    ),
    ("MetaMoTaskSelector", "BaselineCompactQ", "Task-Baseline"),
    ("MetaMoSafetySelector", "BaselineCompactQ", "Safety-Baseline"),
    ("MetaMoComposedSelector", "MetaMoTaskSelector", "Composed-Task"),
    ("MetaMoComposedSelector", "MetaMoSafetySelector", "Composed-Safety"),
    ("MetaMo", "MetaMoComposedSelector", "HardSafety-Composed"),
    ("MetaMo", "BaselineCompactQ", "Full-Baseline"),
    ("MetaMoRule", "MetaMoNeutral", "Rule-Neutral"),
    ("MetaMoRuleNoExternalRisk", "MetaMoRule", "NoRisk-Rule"),
    ("MetaMoRuleNoExternalRisk", "BaselineCompactQ", "NoRisk-Baseline"),
    ("MetaMoNeutral", "BaselineCompactQ", "Neutral-Baseline"),
)
FACTORIAL_CONTRASTS = (
    (
        "MetaMoTrainMetaMoEval",
        "MetaMoTrainQEval",
        "QTrainMetaMoEval",
        "QTrainQEval",
        "Training-Evaluation-Interaction",
    ),
)
METRICS = (
    "total_reward",
    "minerals_collected",
    "completion_rate",
    "unsafe_rate",
    "lava_rate",
    "path_efficiency",
    "survival_rate",
    "appraisal_influence_rate",
    "selector_influence_rate",
    "hard_safety_intervention_rate",
    "q_regret",
    "task_safety_disagreement",
    "compositionality_error",
    "compositionality_median_error",
    "compositionality_p95_error",
    "compositionality_max_error",
    "compositionality_holds_rate",
    "compositionality_goal_error",
    "compositionality_modulator_error",
    "compositionality_action_holds_rate",
)

CONTRAST_FIELDS = (
    "regime",
    "contrast",
    "treatment",
    "control",
    "metric",
    "n_seed_pairs",
    "mean_difference",
    "std_difference",
    "ci95_low",
    "ci95_high",
)


def paired_contrasts(seed_rows: list[dict]) -> list[dict]:
    indexed = {
        (row["regime"], row["variant"], int(row["agent_seed"])): row
        for row in seed_rows
    }
    regimes = sorted({row["regime"] for row in seed_rows}, key=lambda name: REGIME_ORDER[name])
    results: list[dict] = []
    for regime in regimes:
        seeds = sorted({
            int(row["agent_seed"])
            for row in seed_rows
            if row["regime"] == regime
        })
        for treatment, control, contrast in CONTRASTS:
            for metric in METRICS:
                differences = []
                for seed in seeds:
                    treatment_row = indexed.get((regime, treatment, seed))
                    control_row = indexed.get((regime, control, seed))
                    if not treatment_row or not control_row:
                        continue
                    treatment_value = treatment_row.get(metric, "")
                    control_value = control_row.get(metric, "")
                    if treatment_value == "" or control_value == "":
                        continue
                    differences.append(float(treatment_value) - float(control_value))
                if not differences:
                    continue
                stat = _seed_stat(differences)
                results.append({
                    "regime": regime,
                    "contrast": contrast,
                    "treatment": treatment,
                    "control": control,
                    "metric": metric,
                    "n_seed_pairs": len(differences),
                    "mean_difference": stat["mean"],
                    "std_difference": stat["std"],
                    "ci95_low": stat["ci95_low"],
                    "ci95_high": stat["ci95_high"],
                })
        for (
            both,
            train_only,
            eval_only,
            neither,
            contrast,
        ) in FACTORIAL_CONTRASTS:
            for metric in METRICS:
                differences = []
                for seed in seeds:
                    rows = [
                        indexed.get((regime, variant, seed))
                        for variant in (both, train_only, eval_only, neither)
                    ]
                    if any(row is None for row in rows):
                        continue
                    values = [row.get(metric, "") for row in rows]
                    if any(value == "" for value in values):
                        continue
                    both_value, train_value, eval_value, neither_value = (
                        float(value) for value in values
                    )
                    differences.append(
                        both_value - train_value - eval_value + neither_value
                    )
                if not differences:
                    continue
                stat = _seed_stat(differences)
                results.append({
                    "regime": regime,
                    "contrast": contrast,
                    "treatment": f"{both}-{train_only}",
                    "control": f"{eval_only}-{neither}",
                    "metric": metric,
                    "n_seed_pairs": len(differences),
                    "mean_difference": stat["mean"],
                    "std_difference": stat["std"],
                    "ci95_low": stat["ci95_low"],
                    "ci95_high": stat["ci95_high"],
                })
    return results


def analyze(input_dir: Path) -> Path:
    source = input_dir / "gridworld_fair_seed_summary.csv"
    destination = input_dir / "gridworld_ablation_contrasts.csv"
    with source.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    contrasts = paired_contrasts(rows)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTRAST_FIELDS)
        writer.writeheader()
        writer.writerows(contrasts)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    output = analyze(parse_args(argv).input_dir)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
