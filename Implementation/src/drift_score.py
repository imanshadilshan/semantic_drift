"""Core Drift Score logic: CLIP-embedding change in non-target regions between pre- and post-edit images. Pure logic, CPU-friendly, unit-testable."""

import numpy as np


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_drift_score(pre_regions: dict, post_regions: dict, target_region_ids: set, embed_fn) -> float:
    """pre_regions / post_regions: {region_id: crop}. Regions missing from post_regions are treated as unchanged."""
    drift_scores = []
    for region_id, pre_crop in pre_regions.items():
        if region_id in target_region_ids:
            continue
        post_crop = post_regions.get(region_id, pre_crop)
        similarity = cosine_similarity(embed_fn(pre_crop), embed_fn(post_crop))
        drift_scores.append(1 - similarity)

    if not drift_scores:
        raise ValueError("No non-target regions available to score drift over")

    return float(np.mean(drift_scores))


def compute_chain_drift_score(per_step_scores: list[float]) -> float:
    """Cumulative drift across a chain: total unintended change accumulated over all steps."""
    if not per_step_scores:
        raise ValueError("per_step_scores must be non-empty")

    return float(np.sum(per_step_scores))
