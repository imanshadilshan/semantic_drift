"""Drift mitigation strategies: masked conditioning, region-locking, and (stretch) attention-restricted editing.

Both implemented strategies identify the target region the same way drift_score.py does (SAM boxes
on the pre-edit image + CLIP similarity to the instruction), so mitigation and measurement agree on
what counts as "the region the edit was supposed to touch."
"""

from PIL import Image

from .clip_embed import identify_target_regions
from .edit_runner import run_edit
from .segment import crop_regions, get_region_boxes


def _target_box(pre_image: Image.Image, instruction: str) -> tuple[int, int, int, int]:
    boxes = get_region_boxes(pre_image)
    pre_regions = crop_regions(pre_image, boxes)
    target_ids = identify_target_regions(pre_regions, instruction)
    return boxes[next(iter(target_ids))]


def region_locking(pre_image: Image.Image, instruction: str) -> Image.Image:
    """Runs the edit on the full image as normal, then reverts every pixel outside the target
    region back to the pre-edit image — a post-hoc correction of whatever the model produced.
    """
    post_image = run_edit(pre_image, instruction)
    x, y, w, h = _target_box(pre_image, instruction)

    corrected = pre_image.copy()
    corrected.paste(post_image.crop((x, y, x + w, y + h)), (x, y))
    return corrected


def masked_conditioning(pre_image: Image.Image, instruction: str, padding: int = 32) -> Image.Image:
    """Crops down to just the target region (plus padding for context) BEFORE generation, runs the
    edit only on that crop, then pastes the result back — the model never sees, and so can't touch,
    anything outside the crop. A stricter constraint than region_locking, applied at generation time
    rather than after; expected to show visible seams at the crop boundary (H2 in the proposal).
    """
    x, y, w, h = _target_box(pre_image, instruction)
    x0, y0 = max(x - padding, 0), max(y - padding, 0)
    x1, y1 = min(x + w + padding, pre_image.width), min(y + h + padding, pre_image.height)

    crop = pre_image.crop((x0, y0, x1, y1))
    edited_crop = run_edit(crop, instruction).resize(crop.size)

    corrected = pre_image.copy()
    corrected.paste(edited_crop, (x0, y0))
    return corrected


def attention_restricted_editing(image, mask, instruction: str):
    """Stretch goal (Days 20-21) — cross-attention control informed by Prompt-to-Prompt, biasing
    generation toward the target region without hard-masking pixels. Not required for RQ1-RQ3."""
    raise NotImplementedError
