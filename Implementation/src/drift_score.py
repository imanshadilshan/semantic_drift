"""Core Drift Score logic: CLIP-embedding change in non-target regions between pre- and post-edit images. Pure logic, CPU-friendly, unit-testable."""


def compute_drift_score(pre_regions: dict, post_regions: dict, target_region_ids: set, embed_fn):
    raise NotImplementedError


def compute_chain_drift_score(per_step_scores: list[float]):
    raise NotImplementedError
