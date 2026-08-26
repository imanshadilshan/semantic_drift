"""Prepares a JSON payload of base64-encoded images for the human-eval rating artifact
(Day 24-25, Section 5.5 of the proposal): a stratified sample of 15 chains spanning the drift
score range, each with (original, baseline-edited, region_locking-edited) as compressed JPEGs.

Run from Implementation/: python scripts/prepare_human_eval.py
Writes human_eval/rating_data.json (not committed as-is to the artifact — its content gets
embedded directly into the published HTML page).
"""

import base64
import csv
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "results" / "baseline"
MITIGATED_DIR = ROOT / "results" / "mitigated" / "region_locking"
SCORES_PATH = ROOT / "results" / "baseline_drift_scores.csv"
OUTPUT_PATH = ROOT / "human_eval" / "rating_data.json"

N_PER_STRATUM = 5
SEED = 42
JPEG_QUALITY = 82


def to_jpeg_b64(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def select_stratified_sample() -> list[dict]:
    with open(SCORES_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    cum = [r for r in rows if r["step"] == "cumulative" and r["drift_score"] != ""]
    cum.sort(key=lambda r: float(r["drift_score"]))

    n = len(cum)
    low, mid, high = cum[: n // 3], cum[n // 3 : 2 * n // 3], cum[2 * n // 3 :]

    rng = random.Random(SEED)
    return rng.sample(low, N_PER_STRATUM) + rng.sample(mid, N_PER_STRATUM) + rng.sample(high, N_PER_STRATUM)


def main():
    sample = select_stratified_sample()
    chains = []

    for i, row in enumerate(sample, 1):
        stem = Path(row["image_id"]).stem
        chain_type = row["chain_type"]
        baseline_chain_dir = BASELINE_DIR / f"{stem}_{chain_type}"
        mitigated_chain_dir = MITIGATED_DIR / f"{stem}_{chain_type}"

        chains.append(
            {
                "id": f"chain_{i}",
                "image_id": row["image_id"],
                "chain_type": chain_type,
                "baseline_drift_score": float(row["drift_score"]),  # not shown to raters
                "original": to_jpeg_b64(baseline_chain_dir / "step0_original.png"),
                "baseline_edit": to_jpeg_b64(baseline_chain_dir / "step4.png"),
                "mitigated_edit": to_jpeg_b64(mitigated_chain_dir / "step4.png"),
            }
        )
        print(f"[{i}/{len(sample)}] prepared {stem} ({chain_type})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(chains, f)

    total_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\nWrote {len(chains)} chains to {OUTPUT_PATH} ({total_kb:.0f} KB)")


if __name__ == "__main__":
    main()
