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
