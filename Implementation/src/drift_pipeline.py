"""Shared orchestration for scoring a set of saved edit chains (baseline or a mitigation) against
edit_instructions.json. See scripts/compute_baseline_drift.py and scripts/compute_mitigated_drift.py
for the thin entry points that point this at a specific results directory.
"""

import csv
import json
from pathlib import Path

from PIL import Image

from .clip_embed import embed_image, identify_target_regions
from .drift_score import compute_chain_drift_score, compute_drift_score
from .segment import crop_regions, get_region_boxes

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


def score_chain(chain: dict, results_dir: Path) -> list[dict]:
    stem = Path(chain["image_id"]).stem
    chain_dir = results_dir / f"{stem}_{chain['chain_type']}"
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


def load_already_scored(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        return {(row["image_id"], row["chain_type"]) for row in reader if row["step"] == "cumulative"}


def append_rows(rows: list[dict], output_path: Path) -> None:
    is_new = not output_path.exists()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def run(results_dir: Path, output_path: Path, instructions_path: Path) -> None:
    """Scores every chain in instructions_path against results_dir, writing incrementally to
    output_path and skipping chains it already scored (so a crash only costs one chain, and
    re-running after a fix or a Colab disconnect resumes instead of restarting)."""
    with open(instructions_path) as f:
        chains = json.load(f)

    done = load_already_scored(output_path)
    if done:
        print(f"Resuming: {len(done)}/{len(chains)} chains already scored in {output_path}")

    for i, chain in enumerate(chains, 1):
        key = (chain["image_id"], chain["chain_type"])
        stem = Path(chain["image_id"]).stem
        if key in done:
            continue

        rows = score_chain(chain, results_dir)
        append_rows(rows, output_path)

        step_scores = [r["drift_score"] for r in rows if r["step"] != "cumulative" and r["drift_score"] is not None]
        cumulative = rows[-1]["drift_score"]
        cumulative_str = f"{cumulative:.3f}" if cumulative is not None else "N/A"
        print(
            f"[{i}/{len(chains)}] {stem} ({chain['chain_type']}): "
            f"steps={[round(s, 3) for s in step_scores]} cumulative={cumulative_str}"
        )

    print(f"\nDone. Results in {output_path}")
