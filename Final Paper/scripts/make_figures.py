"""Generates the print figures for the IEEE paper from the actual committed results.

Run from Final Paper/: python scripts/make_figures.py
Writes figures/fig2_drift_by_condition.png and figures/fig4_step_position.png.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path(__file__).resolve().parent.parent.parent / "Implementation" / "results"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

# Fixed categorical order (never reassigned by rank) — validated palette, slots 1-3.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def load_cumulative(name: str) -> list[float]:
    with open(ROOT / name, newline="") as f:
        rows = list(csv.DictReader(f))
    return [float(r["drift_score"]) for r in rows if r["step"] == "cumulative" and r["drift_score"] != ""]


def load_step_means(name: str) -> dict[str, float]:
    with open(ROOT / name, newline="") as f:
        rows = list(csv.DictReader(f))
    means = {}
    for step in ("1", "2", "3", "4"):
        vals = [float(r["drift_score"]) for r in rows if r["step"] == step and r["drift_score"] != ""]
        means[step] = sum(vals) / len(vals)
    return means


def fig_drift_by_condition():
    baseline = load_cumulative("baseline_drift_scores.csv")
    region_locking = load_cumulative("region_locking_drift_scores.csv")
    masked_conditioning = load_cumulative("masked_conditioning_drift_scores.csv")

    labels = ["Baseline", "Region-\nLocking", "Masked\nConditioning"]
    means = [sum(baseline) / len(baseline), sum(region_locking) / len(region_locking), sum(masked_conditioning) / len(masked_conditioning)]
    colors = [BLUE, ORANGE, AQUA]

    fig, ax = plt.subplots(figsize=(3.3, 2.6), dpi=300)
    bars = ax.bar(labels, means, color=colors, width=0.58, zorder=3)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=8.2, color=INK)

    ax.set_ylabel("Mean Cumulative Drift Score")
    ax.set_ylim(0, 0.34)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=0)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT_DIR / "fig2_drift_by_condition.png", dpi=300)
    plt.close(fig)


def fig_step_position():
    means = load_step_means("baseline_drift_scores.csv")
    steps = [1, 2, 3, 4]
    vals = [means[str(s)] for s in steps]

    fig, ax = plt.subplots(figsize=(3.3, 2.6), dpi=300)
    ax.plot(steps, vals, color=BLUE, linewidth=2, zorder=3, marker="o", markersize=6, markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=1)

    for s, v in zip(steps, vals):
        ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.2, color=INK)

    ax.set_xlabel("Chain Position (Step)")
    ax.set_ylabel("Mean Drift Score")
    ax.set_xticks(steps)
    ax.set_ylim(0.055, 0.092)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=0)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT_DIR / "fig4_step_position.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_drift_by_condition()
    fig_step_position()
    print("Wrote fig2_drift_by_condition.png and fig4_step_position.png to", OUT_DIR)
