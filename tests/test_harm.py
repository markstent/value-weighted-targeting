"""Harm module: correlation sweep, estimated harm, solver feasibility."""
import numpy as np
import pytest

from core import dgp
from core.estimators import HARM_STREAM, seed_from
from core.policy import target_topk
from study_harm import estimate_harm, simulate, solve_mu, summarise


def _entry(summary, c, sigma, kappa, rule):
    hits = [e for e in summary['cells'] if (e['c_harm'], e['sigma_h'], e['kappa'], e['rule'])
            == (c, sigma, kappa, rule)]
    assert len(hits) == 1
    return hits[0]


def test_c0_no_concentration():
    raw = simulate(reps=10, n_players=5000, c_harms=[0.0], sigma_hs=[0.0], kappas=[0.30])
    e = _entry(summarise(raw, n_boot=2000), 0.0, 0.0, 0.30, 'tauV')
    lo, hi = e['exposure_minus_pop']['ci95']
    assert lo <= 0.0 <= hi


def test_c_neg_reverses():
    raw = simulate(reps=10, n_players=5000, c_harms=[-1.0], sigma_hs=[0.0], kappas=[0.30])
    e = _entry(summarise(raw, n_boot=2000), -1.0, 0.0, 0.30, 'tauV')
    assert e['exposure_minus_pop']['mean'] < 0.0


def test_solver_flags_infeasible():
    rng = np.random.RandomState(0)
    harm_hat = estimate_harm(rng.uniform(0.05, 0.95, 2000), rng, sigma_h=1.0)
    score = rng.normal(size=2000)
    k = 600
    min_mean = np.sort(harm_hat)[:k].mean()
    mu, feasible = solve_mu(score, harm_hat, k, cap=min_mean - 0.01)
    assert feasible is False
    mu, feasible = solve_mu(score, harm_hat, k, cap=min_mean + 0.01)
    assert feasible is True
    assert harm_hat[target_topk(score - mu * harm_hat, k)].mean() <= min_mean + 0.01 + 1e-9


def test_sigma0_matches_true():
    raw = simulate(reps=3, n_players=3000, c_harms=[1.0], sigma_hs=[0.0], kappas=[0.30])
    cell = raw[(1.0, 0.0, 0.30, 'hard cap')]
    assert np.all(np.array(cell['exposure']) <= np.array(cell['cap']) + 1e-6)
    assert not np.any(cell['violation'])


def test_harm_variance_constant_across_c():
    f = dgp.draw_features(200_000, np.random.RandomState(1))
    sds = []
    for c in (-1.0, -0.5, 0.0, 0.5, 1.0):
        h = dgp.draw_harm(f, np.random.RandomState(seed_from(1, HARM_STREAM)), c)
        sds.append(np.std(np.log(h / (1 - h))))
    assert max(sds) - min(sds) < 0.03


def test_realised_corr_follows_c():
    f = dgp.draw_features(50_000, np.random.RandomState(2))
    ltv = 10 * f['weekly_spend'] + 40 * f['login_freq'] + 5 * f['account_age']
    cors = [np.corrcoef(dgp.draw_harm(f, np.random.RandomState(3), c), ltv)[0, 1]
            for c in (-1.0, 0.0, 1.0)]
    assert cors[0] < -0.3 and abs(cors[1]) < 0.02 and cors[2] > 0.3


def test_population_independent_of_c_harm():
    a = dgp.generate_harm(500, np.random.RandomState(5), harm_rng=np.random.RandomState(1), c_harm=1.0)
    b = dgp.generate_harm(500, np.random.RandomState(5), harm_rng=np.random.RandomState(1), c_harm=-1.0)
    for key in ('true_tau', 'true_ltv', 'y_obs', 'treatment'):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.allclose(a['true_harm'], b['true_harm'])


def test_estimate_harm_clips_and_is_identity_at_zero():
    h = np.array([0.0, 0.3, 1.0])
    out = estimate_harm(h, np.random.RandomState(0), 0.0)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[1], 0.3, atol=1e-12)
