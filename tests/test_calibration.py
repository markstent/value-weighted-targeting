"""Calibration at the evaluation budget; AIPW primary, plug-in a tautology."""
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

import core.policy as policy
from core.dgp import generate_main
from core.policy import THETA_GRID, calibrate_theta


def _setup(n=1500, seed=42):
    data = generate_main(n, np.random.RandomState(seed))
    train_idx, _ = train_test_split(np.arange(n), test_size=0.30, random_state=seed)
    return data, train_idx


def _random_est(rs_seed):
    def est(data, est_idx, val_idx, cate_method='tlearner', *, seed, stream):
        rs = np.random.RandomState(rs_seed)
        m = len(val_idx)
        return rs.normal(0.05, 0.1, m), rs.uniform(50, 2000, m), rs.uniform(0.01, 0.99, m)
    return est


@pytest.mark.parametrize('rs_seed', range(5))
def test_plugin_always_identity(rs_seed):
    data, train_idx = _setup()
    cfg = calibrate_theta(data, train_idx, 42, 0.30, objective='plugin',
                          est_func=_random_est(rs_seed))
    assert cfg == THETA_GRID[0]


@pytest.mark.parametrize('kappa', [0.10, 0.50])
def test_uses_eval_kappa(monkeypatch, kappa):
    data, train_idx = _setup()
    ks = []
    real = policy.target_topk

    def spy(scores, k):
        ks.append((len(scores), k))
        return real(scores, k)
    monkeypatch.setattr(policy, 'target_topk', spy)
    for objective in ('plugin', 'aipw'):
        ks.clear()
        calibrate_theta(data, train_idx, 42, kappa, objective=objective)
        n_val = int(round(len(train_idx) * 0.40))
        assert ks and all(n == n_val and k == int(n_val * kappa) for n, k in ks)
        assert len(ks) == len(THETA_GRID)


def test_aipw_deterministic():
    data, train_idx = _setup()
    a = calibrate_theta(data, train_idx, 42, 0.30)
    b = calibrate_theta(data, train_idx, 42, 0.30)
    assert a == b and a in THETA_GRID


def test_unknown_objective_rejected():
    data, train_idx = _setup()
    with pytest.raises(ValueError):
        calibrate_theta(data, train_idx, 42, 0.30, objective='nope')


class _StubClassifier:
    """Records fit/predict row ids (column 0 of X is the row id); constant output.

    Instances are created mu1 first, then mu0, in every fold, so even-numbered
    instances predict 0.7 (mu1) and odd-numbered predict 0.2 (mu0)."""
    instances = []

    def __init__(self, **kwargs):
        self.const = 0.7 if len(_StubClassifier.instances) % 2 == 0 else 0.2
        self.fit_ids = None
        self.pred_ids = None
        _StubClassifier.instances.append(self)

    def fit(self, X, y):
        self.fit_ids = set(X[:, 0].astype(int))
        return self

    def predict_proba(self, X):
        self.pred_ids = set(X[:, 0].astype(int))
        p = np.full(len(X), self.const)
        return np.column_stack([1 - p, p])


def _binary_data(n, seed, tau=0.1):
    rs = np.random.RandomState(seed)
    X = np.column_stack([np.arange(n), rs.normal(size=n)])
    W = rs.binomial(1, 0.5, n)
    Y = rs.binomial(1, 0.3 + tau * W)
    return X, W, Y


@pytest.mark.parametrize('n_folds', [2, 3])
def test_aipw_scores_crossfit_and_closed_form(monkeypatch, n_folds):
    _StubClassifier.instances = []
    monkeypatch.setattr(policy, 'GradientBoostingClassifier', _StubClassifier)
    X, W, Y = _binary_data(400, 1)
    gamma = policy.aipw_scores(X, W, Y, seed=7, n_folds=n_folds)

    insts = _StubClassifier.instances
    assert len(insts) == 2 * n_folds
    predicted = set()
    for m in insts:
        assert m.fit_ids and m.pred_ids and not (m.fit_ids & m.pred_ids)
        predicted |= m.pred_ids
    assert predicted == set(range(400))      # every unit scored out-of-fold

    m1, m0, e = 0.7, 0.2, 0.5
    expected = (m1 - m0) + W / e * (Y - m1) - (1 - W) / (1 - e) * (Y - m0)
    np.testing.assert_allclose(gamma, expected)


def test_aipw_scores_unbiased_for_tau_with_misspecified_outcome(monkeypatch):
    # Constant (wrong) outcome models: only correct IPW terms recover tau.
    class Const(_StubClassifier):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.const = 0.5
    monkeypatch.setattr(policy, 'GradientBoostingClassifier', Const)
    tau = 0.1
    X, W, Y = _binary_data(200_000, 3, tau)
    gamma = policy.aipw_scores(X, W, Y, seed=7)
    se = gamma.std() / np.sqrt(len(gamma))
    assert abs(gamma.mean() - tau) < 4 * se
    assert se < 0.01   # tolerance is tight enough to exclude a sign flip
