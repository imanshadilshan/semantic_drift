"""Loads edit_instructions.json and raw_images/, validating that every referenced image exists."""

import json
from pathlib import Path

from PIL import Image


def load_edit_chains(instructions_path: str, raw_images_dir: str) -> list[dict]:
    with open(instructions_path) as f:
        chains = json.load(f)

    raw_images_dir = Path(raw_images_dir)
    missing = sorted({c["image_id"] for c in chains if not (raw_images_dir / c["image_id"]).exists()})
    if missing:
        preview = missing[:5]
        suffix = "..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{len(missing)} image(s) referenced in {instructions_path} are missing from "
            f"{raw_images_dir}: {preview}{suffix}"
        )

    return chains


def load_image(image_id: str, raw_images_dir: str) -> Image.Image:
    return Image.open(Path(raw_images_dir) / image_id).convert("RGB")
