"""Computes per-step and cumulative Drift Scores over the baseline edit chains in results/baseline/.

For each step: SAM segments the PRE-edit image once, CLIP picks which of those regions the
instruction targets, and drift_score.py scores how much every OTHER region changed between the
pre- and post-edit image (cropped from the same boxes, so it's a true before/after comparison of
the same spatial area). Writes results/baseline_drift_scores.csv incrementally (one chain at a
time), and skips chains already present in that file, so a crash partway through only costs the
one chain being scored when it happened, not the whole batch.

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
FIELDNAMES = ["image_id", "chain_type", "step", "instruction", "drift_score"]


def score_step(pre_image: Image.Image, post_image: Image.Image, instruction: str):
    """Returns a drift score, or None if this step can't be scored (e.g. SAM found too few
    regions in the pre-image to have anything left over once the target region is excluded —
    seen in practice after heavy stylization edits that flatten out texture)."""
    try:
        boxes = get_region_boxes(pre_image)
        pre_regions = crop_regions(pre_image, boxes)
        post_regions = crop_regions(post_image, boxes)
        target_ids = identify_target_regions(pre_regions, instruction)
        return compute_drift_score(pre_regions, post_regions, target_ids, embed_image)
    except ValueError as e:
        print(f"    Warning: step could not be scored ({e}) — recording as skipped")
        return None


def score_chain(chain: dict) -> list[dict]:
    stem = Path(chain["image_id"]).stem
    chain_dir = BASELINE_DIR / f"{stem}_{chain['chain_type']}"
    instructions = chain["instructions"]

    rows = []
    step_scores = []
    pre_image = Image.open(chain_dir / "step0_original.png").convert("RGB")

    for step, instruction in enumerate(instructions, 1):
        post_image = Image.open(chain_dir / f"step{step}.png").convert("RGB")
        score = score_step(pre_image, post_image, instruction)
        if score is not None:
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

    chain_score = compute_chain_drift_score(step_scores) if step_scores else None
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


def load_already_scored() -> set:
    if not OUTPUT_PATH.exists():
        return set()
    with open(OUTPUT_PATH, newline="") as f:
        reader = csv.DictReader(f)
        return {(row["image_id"], row["chain_type"]) for row in reader if row["step"] == "cumulative"}


def append_rows(rows: list[dict]) -> None:
    is_new = not OUTPUT_PATH.exists()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def main():
    with open(INSTRUCTIONS_PATH) as f:
        chains = json.load(f)

    done = load_already_scored()
    if done:
        print(f"Resuming: {len(done)}/{len(chains)} chains already scored in {OUTPUT_PATH}")

    for i, chain in enumerate(chains, 1):
        key = (chain["image_id"], chain["chain_type"])
        stem = Path(chain["image_id"]).stem
        if key in done:
            continue

        rows = score_chain(chain)
        append_rows(rows)

        step_scores = [r["drift_score"] for r in rows if r["step"] != "cumulative" and r["drift_score"] is not None]
        cumulative = rows[-1]["drift_score"]
        cumulative_str = f"{cumulative:.3f}" if cumulative is not None else "N/A"
        print(
            f"[{i}/{len(chains)}] {stem} ({chain['chain_type']}): "
            f"steps={[round(s, 3) for s in step_scores]} cumulative={cumulative_str}"
        )

    print(f"\nDone. Results in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
