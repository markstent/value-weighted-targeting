"""core.stats.paired_ratio_ci: delta-method CI for a ratio of means over paired replications."""

import numpy as np

from core.stats import paired_ratio_ci, paired_test_raw


def test_constant_baseline_reduces_to_the_paired_diff_ci_over_b():
    rng = np.random.RandomState(0)
    d = rng.normal(5.0, 2.0, 60)
    b = np.full(60, 40.0)
    ratio, lo, hi = paired_ratio_ci(d, b)
    ref = paired_test_raw(d, np.zeros(60))  # CI of mean(d)
    assert np.isclose(ratio, d.mean() / 40.0)
    assert np.isclose(lo, ref["ci_95"][0] / 40.0) and np.isclose(hi, ref["ci_95"][1] / 40.0)


def test_matches_a_bootstrap_on_synthetic_correlated_data():
    rng = np.random.RandomState(1)
    n = 200
    b = rng.normal(100.0, 20.0, n)
    d = 0.4 * b + rng.normal(0.0, 6.0, n)  # difference strongly correlated with baseline
    ratio, lo, hi = paired_ratio_ci(d, b)
    idx = rng.randint(0, n, (4000, n))
    boot = d[idx].mean(axis=1) / b[idx].mean(axis=1)
    blo, bhi = np.percentile(boot, [2.5, 97.5])
    assert lo < ratio < hi
    assert abs(lo - blo) < 0.15 * (bhi - blo) and abs(hi - bhi) < 0.15 * (bhi - blo)
    naive_half = 1.96 * d.std(ddof=1) / np.sqrt(n) / b.mean()  # the old formula ignores Cov(D, B)
    assert (hi - lo) / 2 < 0.8 * naive_half  # covariance narrows the interval


def test_single_replication_has_nan_interval():
    r, lo, hi = paired_ratio_ci([3.0], [6.0])
    assert r == 0.5 and np.isnan(lo) and np.isnan(hi)
