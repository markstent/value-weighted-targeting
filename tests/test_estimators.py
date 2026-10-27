"""DR and R CATE learners, params plumbing, and frozen-tuning protocol."""
import numpy as np
import pytest

from core.estimators import (DEFAULT_PARAMS, LEARNERS, estimate_cate,
                             estimate_cate_dr, estimate_cate_r,
                             _estimate_cate_slearner, _estimate_cate_tlearner)

P_FAST = {'max_depth': 2, 'n_estimators': 60, 'learning_rate': 0.1}


def _rct(n, tau_fn, seed, base=0.3):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, (n, 3))
    W = rng.binomial(1, 0.5, n)
    p = np.clip(base + W * tau_fn(X), 0.01, 0.99)
    Y = rng.binomial(1, p)
    return X, W, Y


def test_dr_recovers_constant_tau():
    X, W, Y = _rct(20000, lambda X: 0.1 + 0 * X[:, 0], 0)
    tau = estimate_cate_dr(X[:10000], X[10000:], W[:10000], Y[:10000],
                           P_FAST, seed=1)
    assert abs(tau.mean() - 0.1) < 0.01


def test_r_recovers_linear_tau():
    X, W, Y = _rct(20000, lambda X: 0.2 * X[:, 0], 2)
    tau = estimate_cate_r(X[:10000], X[10000:], W[:10000], Y[:10000],
                          P_FAST, seed=1)
    truth = 0.2 * X[10000:, 0]
    assert np.corrcoef(tau, truth)[0, 1] > 0.9
    assert abs(np.mean(tau - truth)) < 0.02


@pytest.mark.parametrize('fn', [estimate_cate_dr, estimate_cate_r])
def test_dr_r_deterministic_and_seed_sensitive(fn):
    X, W, Y = _rct(1500, lambda X: 0.1 + 0 * X[:, 0], 3)
    a = fn(X[:1000], X[1000:], W[:1000], Y[:1000], P_FAST, seed=5)
    b = fn(X[:1000], X[1000:], W[:1000], Y[:1000], P_FAST, seed=5)
    assert np.array_equal(a, b)
    assert a.shape == (500,)


def test_default_params_reproduce_legacy_learners():
    X, W, Y = _rct(1200, lambda X: 0.2 * X[:, 0], 4)
    legacy_t = _estimate_cate_tlearner(X[:800], X[800:], W[:800], Y[:800])
    via_t = estimate_cate('T', X[:800], X[800:], W[:800], Y[:800], seed=0)
    assert np.array_equal(legacy_t, via_t)
    legacy_s = _estimate_cate_slearner(X[:800], X[800:], W[:800], Y[:800])
    via_s = estimate_cate('S', X[:800], X[800:], W[:800], Y[:800], seed=0)
    assert np.array_equal(legacy_s, via_s)


def test_learners_and_defaults_cover_all_five():
    assert LEARNERS == ('T', 'X', 'S', 'DR', 'R')
    assert set(DEFAULT_PARAMS) == set(LEARNERS)
    for p in DEFAULT_PARAMS.values():
        assert set(p) == {'max_depth', 'n_estimators', 'learning_rate'}
