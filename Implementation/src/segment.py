"""Thin wrapper around Segment Anything (SAM) for zero-shot region segmentation. GPU-preferred, heavy."""

import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from .utils import get_device

_CHECKPOINT_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "data" / ".cache"

_mask_generator = None


def _ensure_checkpoint(model_type: str) -> Path:
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _CHECKPOINT_DIR / f"sam_{model_type}.pth"
    if not path.exists():
        print(f"Downloading SAM {model_type} checkpoint (one-time)...")
        urllib.request.urlretrieve(_CHECKPOINT_URLS[model_type], path)
    return path


def _get_mask_generator(model_type: str = "vit_b"):
    global _mask_generator
    if _mask_generator is None:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        checkpoint = _ensure_checkpoint(model_type)
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(get_device())
        # points_per_side kept modest (default is 32) to stay within free-tier Colab memory/time budgets.
        _mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=16, min_mask_region_area=500)
    return _mask_generator


def segment_image(image: Image.Image, checkpoint: str = "vit_b") -> dict:
    """Returns {region_id: cropped PIL.Image} — one entry per mask SAM finds, cropped to its bounding box."""
    generator = _get_mask_generator(checkpoint)
    image_np = np.array(image.convert("RGB"))
    masks = generator.generate(image_np)

    regions = {}
    for i, mask in enumerate(masks):
        x, y, w, h = mask["bbox"]
        regions[str(i)] = image.crop((x, y, x + w, y + h))
    return regions
