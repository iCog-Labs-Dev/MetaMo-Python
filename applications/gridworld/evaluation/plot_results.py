"""
Plot CSV outputs produced by applications.gridworld.evaluation.runner.

This script writes dependency-free SVG plots, so it works even when matplotlib
is not installed.

Example:

    python -m applications.gridworld.evaluation.plot_results --input-dir eval_results/fair_v1
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path


REGIME_ORDER = ("low", "medium", "high")
VARIANT_ORDER = (
    "BaselineCompactQ",
    "MetaMoTaskSelector",
    "MetaMoSafetySelector",
    "MetaMoComposedSelector",
    "MetaMo",
    "QTrainQEval",
    "MetaMoTrainQEval",
    "QTrainMetaMoEval",
    "MetaMoTrainMetaMoEval",
    "BaselineCompactQRaw",
    "BaselineSafetyPriorQ",
)
VARIANT_LABELS = {
    "BaselineCompactQ": "Baseline RL",
    "BaselineCompactQRaw": "Baseline raw",
    "BaselineSafetyPriorQ": "Safety-prior RL",
    "MetaMo": "MetaMo composed + safety",
    "MetaMoTaskSelector": "MetaMo task selector",
    "MetaMoSafetySelector": "MetaMo safety selector",
    "MetaMoComposedSelector": "MetaMo composed selector",
    "QTrainQEval": "Q train / Q eval",
    "MetaMoTrainQEval": "MetaMo train / Q eval",
    "QTrainMetaMoEval": "Q train / MetaMo eval",
    "MetaMoTrainMetaMoEval": "MetaMo train / MetaMo eval",
}
VARIANT_COLORS = {
    "BaselineCompactQ": "#2563eb",
    "BaselineCompactQRaw": "#64748b",
    "BaselineSafetyPriorQ": "#f59e0b",
    "MetaMo": "#16a34a",
    "MetaMoTaskSelector": "#0ea5e9",
    "MetaMoSafetySelector": "#dc2626",
    "MetaMoComposedSelector": "#14b8a6",
    "QTrainQEval": "#2563eb",
    "MetaMoTrainQEval": "#f59e0b",
    "QTrainMetaMoEval": "#8b5cf6",
    "MetaMoTrainMetaMoEval": "#16a34a",
}


def _load_summary(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing fair-runner summary CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"summary CSV is empty: {path}")
    return rows


def _float(row: dict, key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


def _selected_regimes(rows: list[dict]) -> list[str]:
    present = {row["regime"] for row in rows}
    ordered = [regime for regime in REGIME_ORDER if regime in present]
    return ordered + sorted(present.difference(ordered))


def _selected_variants(rows: list[dict]) -> list[str]:
    present = {row["variant"] for row in rows}
    ordered = [variant for variant in VARIANT_ORDER if variant in present]
    return ordered + sorted(present.difference(ordered))


def _row_map(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["regime"], row["variant"]): row for row in rows}


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _nice_max(value: float) -> float:
    if value <= 1.0:
        return 1.0
    for step in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        top = ((value + step - 1) // step) * step
        if top / max(value, 1.0) <= 1.35:
            return float(top)
    return value * 1.10


def _svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 13,
    fill: str = "#111827",
    anchor: str = "middle",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{_esc(text)}</text>'
    )


def _svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str = "#cbd5e1",
    width: float = 1.0,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"{dash_attr}/>'
    )


def _svg_rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    stroke: str = "none",
    width: float = 0.0,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" '
        f'height="{max(h, 0):.1f}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{width:.1f}"/>'
    )


def _metric_values(
    row_lookup: dict[tuple[str, str], dict],
    regimes: list[str],
    variant: str,
    metric: str,
    scale: float,
) -> list[tuple[float, float, float] | None]:
    values = []
    for regime in regimes:
        row = row_lookup.get((regime, variant))
        if row is None:
            values.append(None)
            continue
        mean = _float(row, f"{metric}_mean") * scale
        lo = _float(row, f"{metric}_ci95_low") * scale
        hi = _float(row, f"{metric}_ci95_high") * scale
        values.append((mean, lo, hi))
    return values


def _draw_axes(
    parts: list[str],
    x: float,
    y: float,
    w: float,
    h: float,
    ymax: float,
    ylabel: str,
    tick_count: int = 4,
) -> None:
    parts.append(_svg_line(x, y + h, x + w, y + h, "#475569", 1.2))
    parts.append(_svg_line(x, y, x, y + h, "#475569", 1.2))
    for idx in range(tick_count + 1):
        value = ymax * idx / tick_count
        ty = y + h - (value / ymax) * h
        parts.append(_svg_line(x, ty, x + w, ty, "#e2e8f0", 0.9, "4 4"))
        parts.append(_svg_text(x - 8, ty + 4, f"{value:.0f}", 11, "#475569", "end"))
    parts.append(_svg_text(x - 50, y + h / 2, ylabel, 12, "#475569", "middle"))


def _draw_grouped_metric(
    parts: list[str],
    rows: list[dict],
    panel: tuple[float, float, float, float],
    metric: str,
    title: str,
    ylabel: str,
    scale: float,
) -> None:
    px, py, pw, ph = panel
    chart_x = px + 70
    chart_y = py + 45
    chart_w = pw - 95
    chart_h = ph - 95
    regimes = _selected_regimes(rows)
    variants = _selected_variants(rows)
    row_lookup = _row_map(rows)

    series = {
        variant: _metric_values(row_lookup, regimes, variant, metric, scale)
        for variant in variants
    }
    all_highs = [
        value[2]
        for values in series.values()
        for value in values
        if value is not None
    ]
    ymax = _nice_max(max(all_highs) * 1.08)

    parts.append(_svg_text(px + pw / 2, py + 22, title, 16, "#0f172a", "middle", "700"))
    _draw_axes(parts, chart_x, chart_y, chart_w, chart_h, ymax, ylabel)

    group_w = chart_w / max(len(regimes), 1)
    bar_w = min(36.0, group_w * 0.72 / max(len(variants), 1))
    for r_idx, regime in enumerate(regimes):
        cx = chart_x + group_w * (r_idx + 0.5)
        parts.append(_svg_text(cx, chart_y + chart_h + 28, regime.title(), 12, "#334155"))
        for v_idx, variant in enumerate(variants):
            value = series[variant][r_idx]
            if value is None:
                continue
            mean, lo, hi = value
            bx = cx - (len(variants) * bar_w) / 2 + v_idx * bar_w + 2
            by = chart_y + chart_h - (mean / ymax) * chart_h
            bh = chart_y + chart_h - by
            color = VARIANT_COLORS.get(variant, "#334155")
            parts.append(_svg_rect(bx, by, bar_w - 4, bh, color, "#111827", 0.6))

            ey1 = chart_y + chart_h - (hi / ymax) * chart_h
            ey2 = chart_y + chart_h - (lo / ymax) * chart_h
            ex = bx + (bar_w - 4) / 2
            parts.append(_svg_line(ex, ey1, ex, ey2, "#111827", 1.1))
            parts.append(_svg_line(ex - 5, ey1, ex + 5, ey1, "#111827", 1.1))
            parts.append(_svg_line(ex - 5, ey2, ex + 5, ey2, "#111827", 1.1))


def save_metric_grid(rows: list[dict], output_dir: Path) -> Path:
    output_path = output_dir / "gridworld_fair_metrics.svg"
    variants = _selected_variants(rows)
    legend_rows = (len(variants) + 2) // 3
    width, height = 1250, 790 + legend_rows * 27
    panels = (
        (40, 70, 550, 320),
        (620, 70, 550, 320),
        (40, 425, 550, 320),
        (620, 425, 550, 320),
    )
    metric_specs = (
        ("total_reward", "Mean Reward", "Reward", 1.0),
        ("completion_rate", "Completion Rate", "Collected / spawned (%)", 100.0),
        ("lava_rate", "Lava Exposure", "Steps in lava (%)", 100.0),
        ("unsafe_rate", "Unsafe-Zone Exposure", "Steps near/in lava (%)", 100.0),
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        _svg_rect(0, 0, width, height, "#ffffff"),
        _svg_text(width / 2, 34, "GridWorld Fair Evaluation", 22, "#0f172a", "middle", "700"),
    ]
    for panel, spec in zip(panels, metric_specs):
        _draw_grouped_metric(parts, rows, panel, *spec)

    for idx, variant in enumerate(variants):
        x = 70 + (idx % 3) * 390
        y = 780 + (idx // 3) * 27
        parts.append(_svg_rect(x, y - 12, 16, 16, VARIANT_COLORS.get(variant, "#334155")))
        parts.append(_svg_text(x + 24, y + 1, VARIANT_LABELS.get(variant, variant), 12, "#334155", "start"))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


def save_reward_safety_frontier(rows: list[dict], output_dir: Path) -> Path:
    output_path = output_dir / "gridworld_reward_safety_frontier.svg"
    width, height = 1260, 620
    chart_x, chart_y = 90, 65
    chart_w, chart_h = 720, 450
    regimes = _selected_regimes(rows)
    variants = _selected_variants(rows)
    row_lookup = _row_map(rows)

    points = []
    for variant in variants:
        for regime in regimes:
            row = row_lookup.get((regime, variant))
            if row is None:
                continue
            points.append(
                (
                    variant,
                    regime,
                    _float(row, "unsafe_rate_mean") * 100.0,
                    _float(row, "total_reward_mean"),
                )
            )

    xmin = min(point[2] for point in points) * 0.92
    xmax = max(point[2] for point in points) * 1.08
    ymin = min(point[3] for point in points) * 0.90
    ymax = max(point[3] for point in points) * 1.06
    if xmax <= xmin:
        xmin -= 1.0
        xmax += 1.0
    if ymax <= ymin:
        ymin -= 1.0
        ymax += 1.0

    def sx(value: float) -> float:
        return chart_x + ((value - xmin) / (xmax - xmin)) * chart_w

    def sy(value: float) -> float:
        return chart_y + chart_h - ((value - ymin) / (ymax - ymin)) * chart_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        _svg_rect(0, 0, width, height, "#ffffff"),
        _svg_text(width / 2, 34, "Reward vs Unsafe-Zone Exposure", 22, "#0f172a", "middle", "700"),
    ]

    parts.append(_svg_line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, "#475569", 1.2))
    parts.append(_svg_line(chart_x, chart_y, chart_x, chart_y + chart_h, "#475569", 1.2))
    for idx in range(5):
        xv = xmin + (xmax - xmin) * idx / 4
        x = sx(xv)
        parts.append(_svg_line(x, chart_y, x, chart_y + chart_h, "#e2e8f0", 0.9, "4 4"))
        parts.append(_svg_text(x, chart_y + chart_h + 24, f"{xv:.0f}", 11, "#475569"))
        yv = ymin + (ymax - ymin) * idx / 4
        y = sy(yv)
        parts.append(_svg_line(chart_x, y, chart_x + chart_w, y, "#e2e8f0", 0.9, "4 4"))
        parts.append(_svg_text(chart_x - 10, y + 4, f"{yv:.0f}", 11, "#475569", "end"))

    for variant in variants:
        path_points = []
        color = VARIANT_COLORS.get(variant, "#334155")
        for regime in regimes:
            row = row_lookup.get((regime, variant))
            if row is None:
                if len(path_points) > 1:
                    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in path_points)
                    parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2.0"/>')
                path_points = []
                continue
            x = sx(_float(row, "unsafe_rate_mean") * 100.0)
            y = sy(_float(row, "total_reward_mean"))
            path_points.append((x, y))
        if len(path_points) > 1:
            d = " ".join(f"{x:.1f},{y:.1f}" for x, y in path_points)
            parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2.0"/>')

    for variant, regime, unsafe, reward in points:
        x, y = sx(unsafe), sy(reward)
        color = VARIANT_COLORS.get(variant, "#334155")
        marker = "circle" if regime == "low" else "rect" if regime == "medium" else "polygon"
        if marker == "circle":
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" stroke="#111827" stroke-width="1"/>')
        elif marker == "rect":
            parts.append(_svg_rect(x - 7, y - 7, 14, 14, color, "#111827", 1))
        else:
            points_attr = f"{x:.1f},{y - 8:.1f} {x - 8:.1f},{y + 7:.1f} {x + 8:.1f},{y + 7:.1f}"
            parts.append(f'<polygon points="{points_attr}" fill="{color}" stroke="#111827" stroke-width="1"/>')
        parts.append(_svg_text(x + 12, y - 8, regime.title(), 11, "#334155", "start"))

    parts.append(_svg_text(chart_x + chart_w / 2, height - 42, "Unsafe-zone exposure (% of steps)", 14, "#334155"))
    parts.append(_svg_text(28, chart_y + chart_h / 2, "Mean reward", 14, "#334155"))

    legend_y = 92
    for idx, variant in enumerate(variants):
        y = legend_y + idx * 28
        parts.append(_svg_rect(900, y - 12, 16, 16, VARIANT_COLORS.get(variant, "#334155")))
        parts.append(_svg_text(923, y + 1, VARIANT_LABELS.get(variant, variant), 12, "#334155", "start"))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


def save_improvement_plot(rows: list[dict], output_dir: Path) -> Path | None:
    regimes = _selected_regimes(rows)
    variants = _selected_variants(rows)
    if "MetaMo" in variants and "BaselineCompactQ" in variants:
        baseline_variant = "BaselineCompactQ"
        reference_variant = "MetaMo"
    elif (
        "MetaMoTrainMetaMoEval" in variants
        and "QTrainQEval" in variants
    ):
        baseline_variant = "QTrainQEval"
        reference_variant = "MetaMoTrainMetaMoEval"
    else:
        return None

    row_lookup = _row_map(rows)
    regimes = [
        regime for regime in regimes
        if (regime, baseline_variant) in row_lookup
        and (regime, reference_variant) in row_lookup
    ]
    if not regimes:
        return None

    output_path = output_dir / "gridworld_metamo_improvement.svg"
    series = {
        "Reward gain": ("#16a34a", []),
        "Lava reduction": ("#0f766e", []),
        "Unsafe reduction": ("#2563eb", []),
    }
    def pct_change(new_value: float, base_value: float) -> float:
        if base_value == 0.0:
            return 0.0
        return ((new_value - base_value) / base_value) * 100.0

    def pct_reduction(new_value: float, base_value: float) -> float:
        if base_value == 0.0:
            return 0.0
        return ((base_value - new_value) / base_value) * 100.0

    for regime in regimes:
        baseline = row_lookup[(regime, baseline_variant)]
        metamo = row_lookup[(regime, reference_variant)]
        base_reward = _float(baseline, "total_reward_mean")
        base_lava = _float(baseline, "lava_rate_mean")
        base_unsafe = _float(baseline, "unsafe_rate_mean")
        series["Reward gain"][1].append(pct_change(_float(metamo, "total_reward_mean"), base_reward))
        series["Lava reduction"][1].append(pct_reduction(_float(metamo, "lava_rate_mean"), base_lava))
        series["Unsafe reduction"][1].append(pct_reduction(_float(metamo, "unsafe_rate_mean"), base_unsafe))

    width, height = 980, 560
    chart_x, chart_y = 85, 65
    chart_w, chart_h = 760, 370
    ymax = _nice_max(max(max(values) for _, values in series.values()) * 1.10)

    def sy(value: float) -> float:
        return chart_y + chart_h - (value / ymax) * chart_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        _svg_rect(0, 0, width, height, "#ffffff"),
        _svg_text(
            width / 2,
            34,
            f"{VARIANT_LABELS.get(reference_variant, reference_variant)} Improvement over Baseline RL",
            22,
            "#0f172a",
            "middle",
            "700",
        ),
    ]
    _draw_axes(parts, chart_x, chart_y, chart_w, chart_h, ymax, "Relative change (%)")

    group_w = chart_w / len(regimes)
    bar_w = min(44, group_w * 0.22)
    for r_idx, regime in enumerate(regimes):
        cx = chart_x + group_w * (r_idx + 0.5)
        parts.append(_svg_text(cx, chart_y + chart_h + 28, regime.title(), 12, "#334155"))
        for s_idx, (label, (color, values)) in enumerate(series.items()):
            bx = cx - 1.5 * bar_w + s_idx * bar_w + 3
            value = values[r_idx]
            by = sy(value)
            parts.append(_svg_rect(bx, by, bar_w - 6, chart_y + chart_h - by, color, "#111827", 0.6))
            parts.append(_svg_text(bx + (bar_w - 6) / 2, by - 6, f"{value:.0f}", 10, "#334155"))

    legend_y = 92
    for idx, (label, (color, _)) in enumerate(series.items()):
        y = legend_y + idx * 28
        parts.append(_svg_rect(870, y - 12, 16, 16, color))
        parts.append(_svg_text(894, y + 1, label, 12, "#334155", "start"))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


def save_plots(input_dir: Path, output_dir: Path | None = None) -> list[Path]:
    summary_path = input_dir / "gridworld_fair_summary.csv"
    output_dir = output_dir or (input_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_summary(summary_path)

    paths = [
        save_metric_grid(rows, output_dir),
        save_reward_safety_frontier(rows, output_dir),
    ]
    improvement_path = save_improvement_plot(rows, output_dir)
    if improvement_path is not None:
        paths.append(improvement_path)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="eval_results/fair_v1")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else None
    paths = save_plots(Path(args.input_dir), output_dir)
    print("Saved plots:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
