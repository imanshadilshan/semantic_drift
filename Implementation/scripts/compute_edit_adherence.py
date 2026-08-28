"""Computes an Edit Adherence Score: CLIP image-text similarity between each chain's final
output and its final instruction, for baseline and both mitigations. Answers a question the
Drift Score alone cannot: does reducing unintended change come at the cost of the requested
edit no longer happening? Whole-image CLIP score (not region-cropped), so it needs no SAM call
and no new GPU run — it runs entirely on the images and instructions already committed locally.

Run from Implementation/: python scripts/compute_edit_adherence.py
Writes results/edit_adherence.csv.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from src.clip_embed import embed_image, embed_text
from src.drift_score import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
INSTRUCTIONS_PATH = ROOT / "data" / "edit_instructions.json"
OUTPUT_PATH = RESULTS_DIR / "edit_adherence.csv"

CONDITIONS = {
    "baseline": RESULTS_DIR / "baseline",
    "region_locking": RESULTS_DIR / "mitigated" / "region_locking",
    "masked_conditioning": RESULTS_DIR / "mitigated" / "masked_conditioning",
}


def main():
    with open(INSTRUCTIONS_PATH) as f:
        chains = json.load(f)

    rows = []
    for i, chain in enumerate(chains, 1):
        stem = Path(chain["image_id"]).stem
        chain_type = chain["chain_type"]
        final_instruction = chain["instructions"][-1]
        text_embed = embed_text(final_instruction)

        pre_image = Image.open(CONDITIONS["baseline"] / f"{stem}_{chain_type}" / "step0_original.png").convert("RGB")
        pre_score = cosine_similarity(embed_image(pre_image), text_embed)

        for condition, base_dir in CONDITIONS.items():
            final_image = Image.open(base_dir / f"{stem}_{chain_type}" / "step4.png").convert("RGB")
            post_score = cosine_similarity(embed_image(final_image), text_embed)
            rows.append(
                {
                    "image_id": chain["image_id"],
                    "chain_type": chain_type,
                    "condition": condition,
                    "instruction": final_instruction,
                    "pre_clip_score": pre_score,
                    "post_clip_score": post_score,
                    "adherence_gain": post_score - pre_score,
                }
            )

        print(f"[{i}/{len(chains)}] {stem} ({chain_type}) done")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
