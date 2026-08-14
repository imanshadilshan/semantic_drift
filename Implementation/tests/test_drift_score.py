"""Unit tests for drift_score.py, using dummy masks and fake embeddings — no real model calls."""

import pytest

from src.drift_score import compute_chain_drift_score, compute_drift_score, cosine_similarity


def identity_embed(x):
    return x


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_zero():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_no_drift_when_non_target_regions_unchanged():
    pre = {"a": [1, 0], "b": [0, 1]}
    post = {"a": [1, 0], "b": [0, 1]}
    score = compute_drift_score(pre, post, target_region_ids=set(), embed_fn=identity_embed)
    assert score == pytest.approx(0.0)


def test_full_drift_when_non_target_region_flips():
    pre = {"a": [1, 0]}
    post = {"a": [-1, 0]}
    score = compute_drift_score(pre, post, target_region_ids=set(), embed_fn=identity_embed)
    assert score == pytest.approx(2.0)


def test_target_region_excluded_from_scoring():
    pre = {"target": [1, 0], "other": [1, 0]}
    post = {"target": [-1, 0], "other": [1, 0]}
    score = compute_drift_score(pre, post, target_region_ids={"target"}, embed_fn=identity_embed)
    assert score == pytest.approx(0.0)


def test_missing_post_region_falls_back_to_pre_crop():
    pre = {"a": [1, 0]}
    post = {}
    score = compute_drift_score(pre, post, target_region_ids=set(), embed_fn=identity_embed)
    assert score == pytest.approx(0.0)


def test_raises_when_no_non_target_regions():
    pre = {"a": [1, 0]}
    post = {"a": [1, 0]}
    with pytest.raises(ValueError):
        compute_drift_score(pre, post, target_region_ids={"a"}, embed_fn=identity_embed)


def test_chain_drift_score_sums_per_step_scores():
    assert compute_chain_drift_score([0.1, 0.2, 0.3]) == pytest.approx(0.6)


def test_chain_drift_score_raises_on_empty_input():
    with pytest.raises(ValueError):
        compute_chain_drift_score([])
