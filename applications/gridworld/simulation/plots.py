"""
Evaluation plot export helpers.
"""

import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "metamo_matplotlib_cache"),
)

import numpy as np


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def _episode_rewards(metrics):
    """Extract per-episode total rewards from a metrics collector."""
    return [ep.total_reward for ep in metrics.episodes]


def _episode_lava_touches(metrics):
    """Extract the number of lava contacts for each episode."""
    return [ep.lava_steps for ep in metrics.episodes]


def _moving_average(values, window=5):
    """Compute a simple moving average over a sequence of values."""
    if len(values) < window:
        return []
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid").tolist()


def _save_reward_plot(baseline_rewards, metamo_rewards, output_dir):
    """Generate and save the reward comparison plot."""
    plt = _load_pyplot()
    path = os.path.join(output_dir, "reward_baseline_vs_metamo.png")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)

    bl_x = np.arange(1, len(baseline_rewards) + 1)
    mm_x = np.arange(1, len(metamo_rewards) + 1)

    ax.plot(
        bl_x,
        baseline_rewards,
        marker="o",
        linewidth=2,
        markersize=4,
        color="#2563eb",
        label="Baseline reward",
    )
    ax.plot(
        mm_x,
        metamo_rewards,
        marker="o",
        linewidth=2,
        markersize=4,
        color="#16a34a",
        label="MetaMo reward",
    )

    for rewards, color, label in (
        (baseline_rewards, "#1d4ed8", "Baseline mean"),
        (metamo_rewards, "#15803d", "MetaMo mean"),
    ):
        if rewards:
            mean_reward = float(np.mean(rewards))
            ax.axhline(
                mean_reward,
                color=color,
                linestyle=":",
                linewidth=1.8,
                label=f"{label}: {mean_reward:.1f}",
            )

    for rewards, color, label in (
        (baseline_rewards, "#93c5fd", "Baseline 5-episode average"),
        (metamo_rewards, "#86efac", "MetaMo 5-episode average"),
    ):
        avg = _moving_average(rewards)
        if avg:
            avg_x = np.arange(5, len(rewards) + 1)
            ax.plot(avg_x, avg, color=color, linewidth=2.2, label=label)

    ax.set_title("Reward Collected Per Evaluation Episode")
    ax.set_xlabel("Evaluation episode")
    ax.set_ylabel("Total reward")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_lava_plot(baseline_lava, metamo_lava, output_dir):
    """Generate and save the lava exposure comparison plot."""
    plt = _load_pyplot()
    path = os.path.join(output_dir, "lava_touches_baseline_vs_metamo.png")
    labels = ["Baseline", "MetaMo"]
    values = [
        float(np.mean(baseline_lava)) if baseline_lava else 0.0,
        float(np.mean(metamo_lava)) if metamo_lava else 0.0,
    ]
    stds = [
        float(np.std(baseline_lava)) if baseline_lava else 0.0,
        float(np.std(metamo_lava)) if metamo_lava else 0.0,
    ]
    totals = [sum(baseline_lava), sum(metamo_lava)]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    bars = ax.bar(
        labels,
        values,
        yerr=stds,
        capsize=8,
        color=["#2563eb", "#16a34a"],
        edgecolor="#111827",
        linewidth=1.0,
    )

    ymax = max(values[i] + stds[i] for i in range(len(values))) if values else 1.0
    y_offset = max(ymax * 0.04, 0.4)
    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"{values[idx]:.2f}/ep\n{totals[idx]} total",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("Agent Touching Lava Region")
    ax.set_ylabel("Lava touches per episode")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(0, ymax + y_offset * 4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_evaluation_plots(baseline_metrics, metamo_metrics, output_dir):
    """
    Save reward and lava-touch plots for completed evaluation episodes.

    Returns a list of generated file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    baseline_rewards = _episode_rewards(baseline_metrics)
    metamo_rewards = _episode_rewards(metamo_metrics)
    baseline_lava = _episode_lava_touches(baseline_metrics)
    metamo_lava = _episode_lava_touches(metamo_metrics)

    return [
        _save_reward_plot(baseline_rewards, metamo_rewards, output_dir),
        _save_lava_plot(baseline_lava, metamo_lava, output_dir),
    ]
