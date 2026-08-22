"""Computes per-step and cumulative Drift Scores over the baseline edit chains in results/baseline/.

See src/drift_pipeline.py for how scoring works. Writes results/baseline_drift_scores.csv.

Run from Implementation/: python scripts/compute_baseline_drift.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drift_pipeline import run

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    run(
        results_dir=ROOT / "results" / "baseline",
        output_path=ROOT / "results" / "baseline_drift_scores.csv",
        instructions_path=ROOT / "data" / "edit_instructions.json",
    )
