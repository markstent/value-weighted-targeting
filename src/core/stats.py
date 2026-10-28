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
    """Two one-sided test for equivalence with margin (all values unrounded)."""
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
        "mean_diff": float(mean_diff),
        "margin": float(margin),
        "p_tost": float(p_tost),
        "equivalent": bool(p_tost < 0.05),
    }


def paired_test_raw(w1, w2, label=None):
    """Paired t-test with effect size and t-interval, all UNROUNDED (the single
    implementation; label is informational only).  A zero-variance difference has
    t_stat NaN and p = 1 if the mean difference is 0 else 0."""
    w1, w2 = np.asarray(w1, dtype=float), np.asarray(w2, dtype=float)
    diff = w1 - w2
    n = len(diff)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    if n > 1 and sd == 0:
        t_stat, p_val = float("nan"), (1.0 if mean == 0 else 0.0)
    else:
        t_stat, p_val = scipy_stats.ttest_rel(w1, w2)
    half = scipy_stats.t.ppf(0.975, df=n - 1) * se if n > 1 else float("nan")
    return {
        "mean_diff": mean,
        "se_diff": float(se),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": (0.0 if sd == 0 else mean / sd) if n > 1 else float("nan"),
        "ci_95": [mean - float(half), mean + float(half)],
    }


def t_mean_ci(x):
    """{'mean', 'ci95'} of a sample: unrounded mean with a t-interval (ci95 None for n < 2)."""
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    if len(x) < 2:
        return {"mean": m, "ci95": None}
    half = scipy_stats.t.ppf(0.975, df=len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {"mean": m, "ci95": [m - half, m + half]}


def paired_ratio_ci(d, b):
    """Ratio of means R = mean(d) / mean(b) over paired replications, with a delta-method
    95% t-interval.  d is the per-replication paired difference, b the per-replication
    baseline.  Var(R) ~ [Var(D) - 2 R Cov(D, B) + R^2 Var(B)] / (n * mean(b)^2), with sample
    (ddof=1) moments, so the baseline's variance and its covariance with the difference are
    kept.  Returns (ratio, lo, hi), all unrounded; NaN interval for n < 2.  When b is
    constant this reduces exactly to the paired-difference CI divided by b."""
    d, b = np.asarray(d, dtype=float), np.asarray(b, dtype=float)
    n = len(d)
    mb = float(b.mean())
    ratio = float(d.mean()) / mb
    if n < 2:
        return ratio, float("nan"), float("nan")
    cov = np.cov(d, b, ddof=1)
    var = (cov[0, 0] - 2 * ratio * cov[0, 1] + ratio**2 * cov[1, 1]) / (n * mb**2)
    half = scipy_stats.t.ppf(0.975, df=n - 1) * np.sqrt(max(var, 0.0))
    return ratio, ratio - float(half), ratio + float(half)


def ratio_entry(d, b):
    """paired_ratio_ci as a JSON-ready {'ratio', 'ci95'} entry."""
    r, lo, hi = paired_ratio_ci(d, b)
    return {"ratio": r, "ci95": [lo, hi]}
