"""Paired significance tests comparing baseline vs. mitigated Drift Scores. Pure logic, CPU-only,
unit-testable — thin wrappers around scipy so the analysis script has one place to call for both
tests rather than importing scipy directly in two places.
"""

from scipy import stats


def paired_t_test(baseline_scores: list[float], mitigated_scores: list[float]):
    """Returns (statistic, p_value). Assumes score differences are roughly normal — use
    wilcoxon_signed_rank instead when that's doubtful, which is the safer default for small,
    likely-skewed samples like this project's (Section 5.5 of the proposal)."""
    if len(baseline_scores) != len(mitigated_scores):
        raise ValueError("baseline_scores and mitigated_scores must be the same length (paired)")
    if len(baseline_scores) < 2:
        raise ValueError("Need at least 2 paired observations to run a t-test")

    result = stats.ttest_rel(baseline_scores, mitigated_scores)
    return float(result.statistic), float(result.pvalue)


def wilcoxon_signed_rank(baseline_scores: list[float], mitigated_scores: list[float]):
    """Returns (statistic, p_value). Non-parametric — doesn't assume normally-distributed
    differences, which is why this is the primary test for this project's small sample size."""
    if len(baseline_scores) != len(mitigated_scores):
        raise ValueError("baseline_scores and mitigated_scores must be the same length (paired)")
    if len(baseline_scores) < 2:
        raise ValueError("Need at least 2 paired observations to run a Wilcoxon test")

    result = stats.wilcoxon(baseline_scores, mitigated_scores)
    return float(result.statistic), float(result.pvalue)
