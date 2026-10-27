"""Feasible (E[tau|X] E[V|X]) oracle versus the hindsight (realised) oracle."""
import numpy as np
import pytest

from core.dgp import (DGPS, FEATURE_NAMES, conditional_means, draw_features,
                      generate_main, structural_main)
from core.policy import compute_welfare, f_churn_vec, target_topk
from simulation_v38 import SEED_BASE


def _features(data):
    return {name: data['features'][:, i] for i, name in enumerate(FEATURE_NAMES)}


def test_feasible_le_hindsight():
    feas, hind = [], []
    for r in range(20):
        seed = SEED_BASE + 7 * r
        data = generate_main(2000, np.random.RandomState(seed))
        cm = conditional_means('main', _features(data), n_mc=100, seed=seed + 10**6)
        k = 600
        tau, ltv = data['true_tau'], data['true_ltv']
        hind.append(compute_welfare(target_topk(tau * ltv, k), tau, ltv))
        feas.append(compute_welfare(
            target_topk(cm['tau_bar'] * cm['ltv_bar'], k), tau, ltv))
    assert np.mean(feas) <= np.mean(hind)


def test_noise_free_equals_hindsight():
    rng = np.random.RandomState(3)
    f = draw_features(1500, rng)
    s = structural_main(f, rng, noise_scale=0.0)
    cm = conditional_means('main', f, n_mc=5, seed=1, noise_scale=0.0)
    np.testing.assert_allclose(cm['tau_bar'], s['true_tau'], atol=1e-12)
    np.testing.assert_allclose(cm['ltv_bar'], s['true_ltv'], atol=1e-9)
    np.testing.assert_allclose(cm['churn_bar'], s['true_churn'], atol=1e-12)
    k = 450
    top_h = set(target_topk(s['true_tau'] * s['true_ltv'], k))
    top_f = set(target_topk(cm['tau_bar'] * cm['ltv_bar'], k))
    assert top_h == top_f


def test_mc_se_small():
    f = draw_features(1000, np.random.RandomState(5))
    cm = conditional_means('main', f, n_mc=400, seed=9)
    assert cm['tau_se'].max() < 0.005


@pytest.mark.parametrize('name', sorted(DGPS))
def test_conditional_means_keys_and_ranges(name):
    f = draw_features(200, np.random.RandomState(2))
    cm = conditional_means(name, f, n_mc=50, seed=4)
    for key in ('tau_bar', 'ltv_bar', 'churn_bar'):
        assert cm[key].shape == (200,) and np.all(np.isfinite(cm[key]))
    assert np.all((cm['churn_bar'] >= 0.01) & (cm['churn_bar'] <= 0.99))
    assert ('harm_bar' in cm) == (name == 'harm')


def test_conditional_means_deterministic_and_independent_of_global_rng():
    f = draw_features(100, np.random.RandomState(2))
    a = conditional_means('main', f, n_mc=40, seed=4)
    np.random.seed(0)
    b = conditional_means('main', f, n_mc=40, seed=4)
    np.testing.assert_array_equal(a['tau_bar'], b['tau_bar'])


def test_chunking_is_statistically_equivalent():
    # Different chunk sizes consume the RNG in a different order, so the draws
    # differ; the estimates must still agree within Monte Carlo error.
    f = draw_features(50, np.random.RandomState(2))
    a = conditional_means('main', f, n_mc=200, seed=4, max_units_per_chunk=10**6)
    b = conditional_means('main', f, n_mc=200, seed=4, max_units_per_chunk=1000)
    tol = 6 * np.sqrt(a['tau_se'] ** 2 + b['tau_se'] ** 2)
    assert np.all(np.abs(a['tau_bar'] - b['tau_bar']) < tol)


def test_simulation_reports_both_oracles():
    from simulation_v38 import run_single_simulation
    res = run_single_simulation(SEED_BASE, n_players=1500)[0]
    for name in ('Oracle hindsight (g=1)', 'Oracle feasible (g=1)',
                 'Oracle feasible (theta0)'):
        assert name in res
    assert 'tauV Oracle' not in res
