"""Computes per-step and cumulative Drift Scores over the baseline edit chains in results/baseline/.

For each step: SAM segments the PRE-edit image once, CLIP picks which of those regions the
instruction targets, and drift_score.py scores how much every OTHER region changed between the
pre- and post-edit image (cropped from the same boxes, so it's a true before/after comparison of
the same spatial area). Writes results/baseline_drift_scores.csv.

Run from Implementation/: python scripts/compute_baseline_drift.py
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from src.clip_embed import embed_image, identify_target_regions
from src.drift_score import compute_chain_drift_score, compute_drift_score
from src.segment import crop_regions, get_region_boxes

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "results" / "baseline"
INSTRUCTIONS_PATH = ROOT / "data" / "edit_instructions.json"
OUTPUT_PATH = ROOT / "results" / "baseline_drift_scores.csv"


def score_chain(chain: dict) -> list[dict]:
    stem = Path(chain["image_id"]).stem
    chain_dir = BASELINE_DIR / f"{stem}_{chain['chain_type']}"
    instructions = chain["instructions"]

    rows = []
    step_scores = []
    pre_image = Image.open(chain_dir / "step0_original.png").convert("RGB")

    for step, instruction in enumerate(instructions, 1):
        post_image = Image.open(chain_dir / f"step{step}.png").convert("RGB")

        boxes = get_region_boxes(pre_image)
        pre_regions = crop_regions(pre_image, boxes)
        post_regions = crop_regions(post_image, boxes)

        target_ids = identify_target_regions(pre_regions, instruction)
        score = compute_drift_score(pre_regions, post_regions, target_ids, embed_image)
        step_scores.append(score)

        rows.append(
            {
                "image_id": chain["image_id"],
                "chain_type": chain["chain_type"],
                "step": step,
                "instruction": instruction,
                "drift_score": score,
            }
        )
        pre_image = post_image

    chain_score = compute_chain_drift_score(step_scores)
    rows.append(
        {
            "image_id": chain["image_id"],
            "chain_type": chain["chain_type"],
            "step": "cumulative",
            "instruction": "",
            "drift_score": chain_score,
        }
    )
    return rows


def main():
    with open(INSTRUCTIONS_PATH) as f:
        chains = json.load(f)

    all_rows = []
    for i, chain in enumerate(chains, 1):
        rows = score_chain(chain)
        all_rows.extend(rows)

        stem = Path(chain["image_id"]).stem
        step_scores = [r["drift_score"] for r in rows if r["step"] != "cumulative"]
        cumulative = rows[-1]["drift_score"]
        print(
            f"[{i}/{len(chains)}] {stem} ({chain['chain_type']}): "
            f"steps={[round(s, 3) for s in step_scores]} cumulative={cumulative:.3f}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "chain_type", "step", "instruction", "drift_score"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
