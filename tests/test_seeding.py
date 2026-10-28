import numpy as np
import pytest

from core.estimators import CALIB_STREAM, LTV_STREAM, _estimate_ltv_churn, seed_from
from core.stats import paired_test_raw, tost_paired


def test_seed_from_formula():
    assert seed_from(42, 0) == (42 * 1_000_003) % 2**31
    assert seed_from(49, 3) == (49 * 1_000_003 + 3) % 2**31
    assert 0 <= seed_from(2**40, 5) < 2**31


def test_streams_are_distinct():
    assert seed_from(42, LTV_STREAM) != seed_from(42, CALIB_STREAM)
    assert seed_from(42, 0) != seed_from(49, 0)


def test_ltv_churn_deterministic_given_seed_and_stream():
    rng = np.random.RandomState(0)
    X = np.column_stack(
        [rng.normal(24, 8, 400), rng.uniform(5, 200, 400), rng.poisson(4, 400) + 1.0, rng.normal(3, 1.5, 400)]
    )
    W = rng.binomial(1, 0.5, 400)
    Y = rng.binomial(1, 0.6, 400)
    a = _estimate_ltv_churn(X[:300], X[300:], W[:300], Y[:300], seed=7)
    b = _estimate_ltv_churn(X[:300], X[300:], W[:300], Y[:300], seed=7)
    c = _estimate_ltv_churn(X[:300], X[300:], W[:300], Y[:300], seed=7, stream=CALIB_STREAM)
    assert np.array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])


def test_paired_test_uses_ddof1():
    x, y = [3.0, 5.0, 9.0, 4.0], [1.0, 2.0, 2.0, 1.0]
    diff = np.array(x) - np.array(y)
    expected = np.std(diff, ddof=1) / np.sqrt(len(diff))
    assert paired_test_raw(x, y)["se_diff"] == pytest.approx(expected, abs=1e-12)


def test_dead_rounded_stats_helpers_are_gone():
    import core.stats as st

    for name in ("paired_test", "cohens_d_paired", "ci_95"):
        assert not hasattr(st, name), name


def test_tost_is_unrounded():
    x, y = [3.1234, 5.4321, 9.9876, 4.5555], [1.0, 2.0, 2.0, 1.0]
    t = tost_paired(x, y, 1.23456)
    assert t["margin"] == 1.23456
    assert t["mean_diff"] == pytest.approx(np.mean(np.array(x) - np.array(y)), abs=1e-15)
    assert t["p_tost"] != round(t["p_tost"], 4)
