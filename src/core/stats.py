"""Paired-comparison statistics (all standard deviations use ddof=1)."""

import numpy as np
from scipy import stats as scipy_stats


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction to a list of p-values."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted = np.ones(n)
    for rank, idx in enumerate(sorted_indices):
        adjusted[idx] = min(1.0, p_values[idx] * (n - rank))
    for i in range(1, n):
        idx = sorted_indices[i]
        prev_idx = sorted_indices[i - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return adjusted


def tost_paired(x, y, margin):
    """Two one-sided test for equivalence with margin."""
    diff = np.array(x) - np.array(y)
    n = len(diff)
    mean_diff = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    t_upper = (mean_diff - margin) / se
    t_lower = (mean_diff + margin) / se
    p_upper = scipy_stats.t.cdf(t_upper, df=n - 1)
    p_lower = 1 - scipy_stats.t.cdf(t_lower, df=n - 1)
    p_tost = max(p_upper, p_lower)
    return {
        'mean_diff': round(float(mean_diff), 0),
        'margin': round(float(margin), 0),
        'p_tost': round(float(p_tost), 4),
        'equivalent': bool(p_tost < 0.05),
    }


def cohens_d_paired(x, y):
    """Cohen's d for paired observations."""
    diff = np.array(x) - np.array(y)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return round(float(np.mean(diff) / sd), 3)


def ci_95(x, y):
    """95% CI for mean difference of paired observations."""
    diff = np.array(x) - np.array(y)
    n = len(diff)
    mean_d = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    margin = scipy_stats.t.ppf(0.975, df=n - 1) * se
    return (round(float(mean_d - margin), 0), round(float(mean_d + margin), 0))


def paired_test(w1, w2, label=None):
    """Paired t-test with effect size and CI (label is informational only)."""
    diff = np.array(w1) - np.array(w2)
    t_stat, p_val = scipy_stats.ttest_rel(w1, w2)
    d = cohens_d_paired(w1, w2)
    lo, hi = ci_95(w1, w2)
    return {
        'mean_diff': round(float(np.mean(diff)), 0),
        'se_diff': round(float(np.std(diff, ddof=1) / np.sqrt(len(diff))), 0),
        't_stat': round(float(t_stat), 2),
        'p_value': round(float(p_val), 4),
        'cohens_d': d,
        'ci_95': [lo, hi],
    }
