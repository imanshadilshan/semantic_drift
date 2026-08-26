"""Unit tests for stats.py using dummy score lists — no real drift data or model calls."""

import pytest

from src.stats import paired_t_test, wilcoxon_signed_rank


def test_paired_t_test_no_systematic_difference_gives_high_p_value():
    # Small, non-systematic noise around the same mean — no real effect to detect.
    baseline = [0.30, 0.31, 0.29, 0.30, 0.32, 0.28]
    mitigated = [0.31, 0.29, 0.30, 0.32, 0.29, 0.30]
    _, p = paired_t_test(baseline, mitigated)
    assert p > 0.05


def test_paired_t_test_consistent_reduction_is_significant():
    baseline = [0.30, 0.32, 0.28, 0.35, 0.29, 0.31]
    mitigated = [0.05, 0.06, 0.04, 0.07, 0.05, 0.06]
    stat, p = paired_t_test(baseline, mitigated)
    assert p < 0.05
    assert stat > 0  # baseline consistently higher than mitigated


def test_paired_t_test_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_t_test([0.1, 0.2], [0.1])


def test_paired_t_test_raises_on_too_few_observations():
    with pytest.raises(ValueError):
        paired_t_test([0.1], [0.2])


def test_wilcoxon_consistent_reduction_is_significant():
    baseline = [0.30, 0.32, 0.28, 0.35, 0.29, 0.31]
    mitigated = [0.05, 0.06, 0.04, 0.07, 0.05, 0.06]
    _, p = wilcoxon_signed_rank(baseline, mitigated)
    assert p < 0.05


def test_wilcoxon_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        wilcoxon_signed_rank([0.1, 0.2], [0.1])


def test_wilcoxon_raises_on_too_few_observations():
    with pytest.raises(ValueError):
        wilcoxon_signed_rank([0.1], [0.2])
