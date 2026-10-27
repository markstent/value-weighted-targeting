import numpy as np
import pytest

from core.estimators import (CALIB_STREAM, LTV_STREAM, _estimate_ltv_churn,
                             seed_from)
from core.stats import paired_test


def test_seed_from_formula():
    assert seed_from(42, 0) == (42 * 1_000_003) % 2**31
    assert seed_from(49, 3) == (49 * 1_000_003 + 3) % 2**31
    assert 0 <= seed_from(2**40, 5) < 2**31


def test_streams_are_distinct():
    assert seed_from(42, LTV_STREAM) != seed_from(42, CALIB_STREAM)
    assert seed_from(42, 0) != seed_from(49, 0)


def test_ltv_churn_deterministic_given_seed_and_stream():
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.normal(24, 8, 400), rng.uniform(5, 200, 400),
                         rng.poisson(4, 400) + 1.0, rng.normal(3, 1.5, 400)])
    W = rng.binomial(1, 0.5, 400)
    Y = rng.binomial(1, 0.6, 400)
    a = _estimate_ltv_churn(X[:300], X[300:], W[:300], Y[:300], seed=7)
    b = _estimate_ltv_churn(X[:300], X[300:], W[:300], Y[:300], seed=7)
    c = _estimate_ltv_churn(X[:300], X[300:], W[:300], Y[:300], seed=7,
                            stream=CALIB_STREAM)
    assert np.array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])


def test_paired_test_uses_ddof1():
    x, y = [3.0, 5.0, 9.0, 4.0], [1.0, 2.0, 2.0, 1.0]
    diff = np.array(x) - np.array(y)
    expected = np.std(diff, ddof=1) / np.sqrt(len(diff))
    assert paired_test(x, y)['se_diff'] == pytest.approx(round(expected), abs=1e-9)
