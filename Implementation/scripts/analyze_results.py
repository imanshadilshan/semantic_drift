"""Day 22-23 statistical analysis: paired significance tests comparing baseline vs. each
mitigation strategy, plus a chain-type breakdown (RQ3) and a step-position breakdown (RQ2/H1).

Pure CPU work over the already-computed CSVs — no models, no GPU. Run from Implementation/:
python scripts/analyze_results.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats import paired_t_test, wilcoxon_signed_rank

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def load(name: str) -> list[dict]:
    with open(RESULTS_DIR / name, newline="") as f:
        return list(csv.DictReader(f))


def cumulative_by_key(rows: list[dict], chain_type: str | None = None) -> dict:
    out = {}
    for r in rows:
        if r["step"] != "cumulative" or r["drift_score"] == "":
            continue
        if chain_type is not None and r["chain_type"] != chain_type:
            continue
        out[(r["image_id"], r["chain_type"])] = float(r["drift_score"])
    return out


def step_scores_by_key(rows: list[dict], step: str) -> dict:
    out = {}
    for r in rows:
        if r["step"] != step or r["drift_score"] == "":
            continue
        out[(r["image_id"], r["chain_type"])] = float(r["drift_score"])
    return out


def paired_values(a: dict, b: dict) -> tuple[list[float], list[float]]:
    """Only keeps chains present with a valid score on BOTH sides — some baseline steps were
    skipped (sparse-region edge case), so a naive zip would silently misalign pairs."""
    common = sorted(set(a) & set(b))
    return [a[k] for k in common], [b[k] for k in common]


def report_comparison(label: str, a_scores: list[float], b_scores: list[float]) -> None:
    n = len(a_scores)
    if n < 2:
        print(f"{label}: not enough paired chains (n={n}) to test")
        return

    mean_a = sum(a_scores) / n
    mean_b = sum(b_scores) / n
    t_stat, t_p = paired_t_test(a_scores, b_scores)
    try:
        w_stat, w_p = wilcoxon_signed_rank(a_scores, b_scores)
        w_str = f"W={w_stat:.1f} p={w_p:.4f}"
    except ValueError as e:
        w_str = f"n/a ({e})"

    reduction = (1 - mean_b / mean_a) * 100 if mean_a else float("nan")
    print(f"{label} (n={n})")
    print(f"  mean: {mean_a:.4f} -> {mean_b:.4f}  ({reduction:+.1f}%)")
    print(f"  paired t-test:      t={t_stat:.3f} p={t_p:.4f}")
    print(f"  Wilcoxon signed-rank: {w_str}")
    print()


def main():
    baseline = load("baseline_drift_scores.csv")
    region_locking = load("region_locking_drift_scores.csv")
    masked_conditioning = load("masked_conditioning_drift_scores.csv")

    print("=" * 70)
    print("RQ3: Does each mitigation significantly reduce cumulative drift?")
    print("=" * 70)
    base_cum = cumulative_by_key(baseline)
    rl_cum = cumulative_by_key(region_locking)
    mc_cum = cumulative_by_key(masked_conditioning)

    report_comparison("Baseline vs region_locking", *paired_values(base_cum, rl_cum))
    report_comparison("Baseline vs masked_conditioning", *paired_values(base_cum, mc_cum))
    report_comparison("region_locking vs masked_conditioning", *paired_values(rl_cum, mc_cum))

    print("=" * 70)
    print("RQ3 breakdown by chain type (object-level vs. global)")
    print("=" * 70)
    for chain_type in ("object_level", "global"):
        base_t = cumulative_by_key(baseline, chain_type)
        rl_t = cumulative_by_key(region_locking, chain_type)
        mc_t = cumulative_by_key(masked_conditioning, chain_type)
        report_comparison(f"[{chain_type}] Baseline vs region_locking", *paired_values(base_t, rl_t))
        report_comparison(f"[{chain_type}] Baseline vs masked_conditioning", *paired_values(base_t, mc_t))

    print("=" * 70)
    print("RQ2/H1: Does drift increase across chain position (compounding)?")
    print("=" * 70)
    steps = {s: step_scores_by_key(baseline, s) for s in ("1", "2", "3", "4")}
    for s in ("1", "2", "3", "4"):
        vals = list(steps[s].values())
        print(f"  step {s}: mean={sum(vals) / len(vals):.4f} (n={len(vals)})")
    print()
    for a, b in (("1", "2"), ("2", "3"), ("3", "4"), ("1", "4")):
        report_comparison(f"Baseline step {a} vs step {b}", *paired_values(steps[a], steps[b]))


if __name__ == "__main__":
    main()
