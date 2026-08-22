"""Computes per-step and cumulative Drift Scores for a mitigation strategy's edit chains.

See src/drift_pipeline.py for how scoring works. Writes results/<strategy>_drift_scores.csv.

Run from Implementation/: python scripts/compute_mitigated_drift.py region_locking
                       or: python scripts/compute_mitigated_drift.py masked_conditioning
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drift_pipeline import run

ROOT = Path(__file__).resolve().parent.parent
STRATEGIES = ("region_locking", "masked_conditioning")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STRATEGIES:
        print(f"Usage: python {Path(__file__).name} <{'|'.join(STRATEGIES)}>")
        sys.exit(1)

    strategy = sys.argv[1]
    run(
        results_dir=ROOT / "results" / "mitigated" / strategy,
        output_path=ROOT / "results" / f"{strategy}_drift_scores.csv",
        instructions_path=ROOT / "data" / "edit_instructions.json",
    )
