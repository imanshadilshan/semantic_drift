"""CLIP image/text embeddings: the embed_fn for drift_score.py, plus target-region identification.
GPU-preferred, heavy.
"""

import numpy as np
import torch
from PIL import Image

from .utils import get_device

_model = None
_processor = None


def _get_clip(checkpoint: str = "openai/clip-vit-base-patch32"):
    global _model, _processor
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor

        _model = CLIPModel.from_pretrained(checkpoint).to(get_device()).eval()
        _processor = CLIPProcessor.from_pretrained(checkpoint)
    return _model, _processor


def embed_image(crop: Image.Image, checkpoint: str = "openai/clip-vit-base-patch32") -> np.ndarray:
    model, processor = _get_clip(checkpoint)
    inputs = processor(images=crop, return_tensors="pt").to(get_device())
    with torch.no_grad():
        # transformers>=5 returns a BaseModelOutputWithPooling; .pooler_output is the projected
        # 512-dim embedding (not the raw 768-dim vision hidden state — last_hidden_state is that).
        features = model.get_image_features(**inputs).pooler_output
    return features[0].cpu().numpy()


def embed_text(text: str, checkpoint: str = "openai/clip-vit-base-patch32") -> np.ndarray:
    model, processor = _get_clip(checkpoint)
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(get_device())
    with torch.no_grad():
        features = model.get_text_features(**inputs).pooler_output
    return features[0].cpu().numpy()


def identify_target_regions(regions: dict, instruction: str, top_k: int = 1) -> set:
    """Returns the region_id(s) whose CLIP embedding is most similar to the instruction text.

    regions should come from the PRE-edit image (see segment.get_region_boxes). This works well
    for "remove"/"change" instructions, where the target already exists pre-edit. It's a weaker
    signal for "add" instructions, where the target only appears post-edit and has no strong match
    among the pre-edit regions — a known limitation, not a bug (see PROGRESS.md).
    """
    from .drift_score import cosine_similarity

    text_embed = embed_text(instruction)
    similarities = {region_id: cosine_similarity(embed_image(crop), text_embed) for region_id, crop in regions.items()}
    ranked = sorted(similarities, key=similarities.get, reverse=True)
    return set(ranked[:top_k])
