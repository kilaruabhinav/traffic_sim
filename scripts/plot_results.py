"""Generate publication-ready figures from a saved traffic evaluation report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "traffic-sim-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONTROLLERS = ("Random", "Rule-based", "DQN", "PPO")
COLORS = ("#6b7280", "#475569", "#2563eb", "#059669")


def load_report(path: Path) -> dict[str, Any]:
    """Load and minimally validate the evaluator's JSON schema."""
    report = json.loads(path.read_text())
    required = {"scenarios", "episode_results", "summaries", "ppo_minus_dqn"}
    missing = required.difference(report)
    if missing:
        raise ValueError(f"evaluation report is missing keys: {sorted(missing)}")
    if not report["episode_results"]:
        raise ValueError("evaluation report contains no episode results")
    return report


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#d1d5db",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight")
    plt.close(fig)


def _group_values(report: dict[str, Any], field: str) -> list[np.ndarray]:
    rows = report["episode_results"]
    return [
        np.asarray([row[field] for row in rows if row["controller"] == controller], dtype=float)
        for controller in CONTROLLERS
    ]


def _comparison_plot(
    report: dict[str, Any],
    output_dir: Path,
    field: str,
    title: str,
    ylabel: str,
    stem: str,
    show_sd: bool = True,
) -> None:
    grouped = _group_values(report, field)
    means = np.asarray([values.mean() for values in grouped])
    errors = np.asarray([values.std(ddof=1) for values in grouped]) if show_sd else None
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.bar(CONTROLLERS, means, color=COLORS, width=0.66, zorder=3)
    if errors is not None:
        ax.errorbar(
            np.arange(len(CONTROLLERS)), means, yerr=errors, fmt="none",
            ecolor="#111827", elinewidth=1.2, capsize=5, label="±1 sample SD", zorder=4,
        )
        ax.legend(frameon=False, loc="best")
    if np.any(means < 0) and np.any(means > 0):
        ax.axhline(0, color="#374151", linewidth=0.9)
    ax.set_title(title, loc="left", weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Controller")
    ax.grid(axis="x", visible=False)
    padding = max(np.ptp(means) * 0.025, max(abs(means)) * 0.015, 0.1)
    for bar, value in zip(bars, means):
        vertical = padding if value >= 0 else -padding
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + vertical, f"{value:,.2f}",
            ha="center", va="bottom" if value >= 0 else "top", fontsize=9,
        )
    _save(fig, output_dir, stem)


def _distribution_plot(
    report: dict[str, Any], output_dir: Path, field: str, title: str, ylabel: str, stem: str
) -> None:
    grouped = _group_values(report, field)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    box = ax.boxplot(grouped, tick_labels=CONTROLLERS, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    for median in box["medians"]:
        median.set_color("white")
        median.set_linewidth(1.8)
    ax.set_title(title, loc="left", weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Controller")
    ax.text(
        0.99, 0.02, "Box: IQR · whiskers: 1.5×IQR · outliers hidden",
        transform=ax.transAxes, ha="right", va="bottom", color="#4b5563", fontsize=8.5,
    )
    ax.grid(axis="x", visible=False)
    _save(fig, output_dir, stem)


def _paired_plot(report: dict[str, Any], output_dir: Path) -> None:
    labels = {
        "reward": ("Reward", "reward per episode"),
        "waiting": ("Waiting", "mean aggregate waiting state"),
        "vehicles_passed": ("Vehicles passed", "vehicles per episode"),
        "violations": ("Violations", "lane-step violations per episode"),
    }
    by_metric = {item["metric"]: item for item in report["ppo_minus_dqn"]}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    for ax, (metric, (title, unit)) in zip(axes.flat, labels.items()):
        item = by_metric[metric]
        mean = float(item["mean"])
        low = float(item["ci95_low"])
        high = float(item["ci95_high"])
        ax.errorbar(
            mean, 0, xerr=[[mean - low], [high - mean]], fmt="o", color="#2563eb",
            ecolor="#2563eb", capsize=6, markersize=7, linewidth=1.8,
        )
        ax.axvline(0, color="#6b7280", linewidth=1)
        ax.set_yticks([])
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel(f"PPO − DQN ({unit})")
        ax.text(
            0.5, 0.68, f"{mean:+.2f}  [95% CI {low:+.2f}, {high:+.2f}]",
            transform=ax.transAxes, ha="center", va="center", fontsize=9,
        )
        ax.set_ylim(-0.22, 0.30)
        span = max(abs(mean - low), abs(high - mean), abs(mean) * 0.2, 0.5)
        ax.set_xlim(min(0, low) - span, max(0, high) + span)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Paired PPO − DQN Differences Across Shared Scenarios", x=0.06, ha="left", weight="bold")
    _save(fig, output_dir, "ppo_vs_dqn_paired_differences")


def create_plots(report: dict[str, Any], output_dir: Path) -> list[Path]:
    """Create all comparison and distribution plots, returning generated paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()
    comparisons = (
        ("reward", "Mean Episode Reward", "episode reward (mean ± 1 SD)", "controller_reward_comparison", True),
        ("total_waiting", "Mean Aggregate Waiting State", "waiting metric (mean ± 1 SD)", "waiting_time_comparison", True),
        ("vehicles_passed", "Episode Throughput", "vehicles passed (mean ± 1 SD)", "throughput_comparison", True),
        ("max_queue", "Maximum Queue per Episode", "vehicles in longest queue (mean ± 1 SD)", "max_queue_comparison", True),
        ("violations", "Soft Queue-Limit Violations", "lane-step violations (mean ± 1 SD)", "violation_comparison", True),
    )
    for field, title, ylabel, stem, show_sd in comparisons:
        _comparison_plot(report, output_dir, field, title, ylabel, stem, show_sd)
    _paired_plot(report, output_dir)
    distributions = (
        ("reward", "Episode Reward Distribution", "episode reward", "reward_distribution"),
        ("total_waiting", "Waiting-Metric Distribution", "mean aggregate waiting state", "waiting_distribution"),
        ("max_queue", "Maximum-Queue Distribution", "vehicles in longest queue", "max_queue_distribution"),
    )
    for field, title, ylabel, stem in distributions:
        _distribution_plot(report, output_dir, field, title, ylabel, stem)
    return sorted(output_dir.glob("*.png")) + sorted(output_dir.glob("*.svg"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/evaluation_500.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/results"))
    args = parser.parse_args()
    generated = create_plots(load_report(args.input), args.output_dir)
    print(f"Generated {len(generated)} plot files in {args.output_dir}")


if __name__ == "__main__":
    main()
