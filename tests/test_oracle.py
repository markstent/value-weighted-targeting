"""Feasible (E[tau|X] E[V|X]) oracle versus the hindsight (realised) oracle."""
import numpy as np
import pytest

from core.dgp import (DGPS, FEATURE_NAMES, conditional_means, draw_features,
                      generate_main, make_corr_dgp, structural_main, structural_null,
                      tau_base_stats)
from core.policy import compute_welfare, f_churn_vec, target_topk
from study_main import SEED_BASE


def _features(data):
    return {name: data['features'][:, i] for i, name in enumerate(FEATURE_NAMES)}


def test_feasible_le_hindsight_sanity():
    # Sanity only: hindsight is the exact top-k maximiser, so this cannot fail
    # for any score; the real checks are the closed-form and plug-in tests below.
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
    from study_main import run_single_simulation
    res = run_single_simulation(SEED_BASE, n_players=1500)[0]
    for name in ('Oracle hindsight (g=1)', 'Oracle feasible (g=1)',
                 'Oracle feasible (theta0)'):
        assert name in res
    assert 'tauV Oracle' not in res


def test_null_tau_bar_matches_closed_form():
    # null DGP: tau = clip(tau_base + N(0, 0.03)); away from the clip bounds
    # E[tau|X] = tau_base.
    rng = np.random.RandomState(11)
    f = draw_features(3000, rng)
    tb = structural_null(f, rng, noise_scale=0.0)['tau_base']
    keep = (tb > -0.10) & (tb < 0.30)
    assert keep.sum() > 500
    sub = {k: v[keep][:300] for k, v in f.items()}
    cm = conditional_means('null', sub, n_mc=5000, seed=3)
    assert np.all(np.abs(cm['tau_bar'] - tb[keep][:300]) < 5 * cm['tau_se'] + 1e-4)
    assert np.abs(cm['tau_bar'] - tb[keep][:300]).max() < 0.002


def test_heavytail_ltv_bar_matches_closed_form():
    # ltv = base_v * exp(N(0,1)) clipped; away from the clips E = base_v * e^0.5.
    f = draw_features(4000, np.random.RandomState(12))
    base_v = (10 * f['weekly_spend'] + 40 * f['login_freq'] + 5 * f['account_age'])
    keep = np.where((base_v > 1000) & (base_v < 3000))[0][:80]
    assert len(keep) == 80
    sub = {k: v[keep] for k, v in f.items()}
    cm = conditional_means('heavytail', sub, n_mc=20000, seed=3)
    rel = cm['ltv_bar'] / (base_v[keep] * np.exp(0.5)) - 1
    assert np.abs(rel).max() < 0.05   # MC rel. SE ~0.9%


def test_feasible_beats_noise_free_plugin():
    # E[tau V|X] is what an X-measurable rule should rank on; ranking on the
    # noise-free plug-in (structural at noise_scale=0) is biased by Jensen terms.
    diffs = []
    for r in range(12):
        seed = SEED_BASE + 7 * r
        rng = np.random.RandomState(seed)
        data = generate_main(3000, rng)
        f = _features(data)
        plug = structural_main(f, np.random.RandomState(0), noise_scale=0.0)
        cm = conditional_means('main', f, n_mc=400, seed=seed + 1)
        k = 900
        tau, ltv = data['true_tau'], data['true_ltv']
        w_f = compute_welfare(target_topk(cm['tau_bar'] * cm['ltv_bar'], k), tau, ltv)
        w_p = compute_welfare(target_topk(plug['true_tau'] * plug['true_ltv'], k), tau, ltv)
        diffs.append(w_f - w_p)
    diffs = np.array(diffs)
    tol = 2 * diffs.std(ddof=1) / np.sqrt(len(diffs))
    assert diffs.mean() >= -tol


def test_corr_gamma_uses_generating_sample_moments_and_gamma():
    gamma = 1.0
    data = make_corr_dgp(gamma)(3000, np.random.RandomState(21))
    full = _features(data)
    stats_full = tau_base_stats(full)
    test = {k: v[:600] for k, v in full.items()}
    stats_sub = tau_base_stats(test)
    assert stats_full != stats_sub
    cm = conditional_means('corr_gamma', test, n_mc=50, seed=1, gamma=gamma,
                           tau_base_stats=stats_full, noise_scale=0.0)
    resid = data['true_ltv'][:600] - cm['ltv_bar']
    # Noise-free ltv_bar must reproduce the DGP up to its N(0, 80) noise (no shift).
    assert abs(resid.mean()) < 4 * 80 / np.sqrt(600)
    assert 60 < resid.std() < 100
    # Wrong gamma is detectable: gamma=0 leaves a systematic gamma-sized residual.
    cm0 = conditional_means('corr_gamma', test, n_mc=50, seed=1, gamma=0.0,
                            tau_base_stats=stats_full, noise_scale=0.0)
    assert (data['true_ltv'][:600] - cm0['ltv_bar']).std() > resid.std() * 1.3


def test_simulation_threads_gamma_and_rejects_unknown_dgp():
    from study_main import run_single_simulation
    res = run_single_simulation(SEED_BASE, n_players=1200,
                                dgp_func=make_corr_dgp(1.0))[0]
    assert 'Oracle feasible (g=1)' in res
    with pytest.raises(ValueError):
        run_single_simulation(SEED_BASE, n_players=1200,
                              dgp_func=lambda n, rng: generate_main(n, rng))
