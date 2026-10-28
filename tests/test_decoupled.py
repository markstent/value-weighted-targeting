"""Decoupled harm DGP: tau_base without the z_spend term, its feasible oracle, and the
extra soft-minus-CATE-only paired test in every harm cell."""
import numpy as np

from core import dgp
from core.dgp import FEATURE_NAMES, conditional_means, draw_features
from core.stats import holm_bonferroni, paired_ratio_ci, paired_test_raw
from study_harm import simulate, summarise


def test_tau_base_has_no_spend_term():
    f = draw_features(2000, np.random.RandomState(1))
    f2 = dict(f, weekly_spend=f['weekly_spend'] * 1.7)        # change spend only
    a = dgp.structural_decoupled(f, np.random.RandomState(0), noise_scale=0.0)
    b = dgp.structural_decoupled(f2, np.random.RandomState(0), noise_scale=0.0)
    np.testing.assert_array_equal(a['tau_base'], b['tau_base'])
    m = dgp.structural_main(f, np.random.RandomState(0), noise_scale=0.0)
    z_spend = (f['weekly_spend'] - dgp.POP_MEAN['weekly_spend']) / dgp.POP_SD['weekly_spend']
    np.testing.assert_allclose(m['tau_base'] - a['tau_base'], -0.04 * z_spend, atol=1e-12)
    # V, churn and baseline are untouched
    np.testing.assert_array_equal(m['true_ltv'], a['true_ltv'])
    np.testing.assert_array_equal(m['y0_prob'], a['y0_prob'])


def test_generate_harm_decoupled_only_changes_tau():
    kw = dict(harm_rng=np.random.RandomState(1))
    main = dgp.generate_harm(500, np.random.RandomState(5), **kw)
    dec = dgp.generate_harm(500, np.random.RandomState(5), harm_rng=np.random.RandomState(1),
                            decoupled=True)
    np.testing.assert_array_equal(main['features'], dec['features'])
    for key in ('true_ltv', 'true_harm', 'treatment', 'true_churn'):
        np.testing.assert_array_equal(main[key], dec[key])
    assert not np.allclose(main['true_tau'], dec['true_tau'])


def test_decoupled_oracle_closed_form_noise_free():
    f = draw_features(800, np.random.RandomState(3))
    cm = conditional_means('main_decoupled', f, n_mc=4, seed=1, noise_scale=0.0)
    z_login = (f['login_freq'] - dgp.POP_MEAN['login_freq']) / dgp.POP_SD['login_freq']
    tau_base = 0.06 + 0.10 * z_login - 0.02 * z_login ** 2
    s = dgp.structural_decoupled(f, np.random.RandomState(0), noise_scale=0.0)
    mod = dgp.churn_moderation_sym(s['true_churn'])
    np.testing.assert_allclose(cm['tau_bar'], (tau_base * mod).clip(-0.2, 0.4), atol=1e-12)


def test_decoupled_oracle_matches_brute_force_monte_carlo():
    f = draw_features(6, np.random.RandomState(4))
    cm = conditional_means('harm_decoupled', f, n_mc=2000, seed=7)
    assert 'harm_bar' in cm
    n_rep = 4000
    rep = {k: np.repeat(v, n_rep) for k, v in f.items()}
    s = dgp.structural_decoupled(rep, np.random.RandomState(99))
    brute = s['true_tau'].reshape(6, n_rep).mean(axis=1)
    tol = 5 * np.sqrt(cm['tau_se'] ** 2 + s['true_tau'].reshape(6, n_rep).std(axis=1) ** 2 / n_rep)
    assert np.all(np.abs(cm['tau_bar'] - brute) < tol)


def test_simulate_decoupled_and_cate_only_test_in_every_cell():
    kw = dict(reps=4, n_players=3000, c_harms=[1.0], sigma_hs=[0.0, 0.5], kappas=[0.30])
    summaries = {}
    for name in ('main', 'decoupled'):
        raw = simulate(dgp=name, **kw)
        s = summarise(raw, n_boot=50, dgp=name)
        assert {c['dgp'] for c in s['cells']} == {name}
        tests = s['paired_tests']
        assert {t['dgp'] for t in tests} == {name}
        assert len(tests) == 2 * 7                       # 2 sigma cells x (6 + 1)
        for sigma in (0.0, 0.5):
            fam = [t for t in tests if t['sigma_h'] == sigma]
            assert len(fam) == 7 and all(t['family_size'] == 7 for t in fam)
            assert [t['p_holm'] for t in fam] == list(holm_bonferroni([t['p_raw'] for t in fam]))
            new = [t for t in fam if t['comparison'] == 'soft gate vs CATE only']
            assert len(new) == 1 and new[0]['components'] == 'estimated'
            t = new[0]
            wa = np.array(raw[(1.0, sigma, 0.30, 'soft gate')]['welfare'])
            wb = np.array(raw[(1.0, sigma, 0.30, 'CATE only')]['welfare'])
            ref = paired_test_raw(wa, wb)
            assert t['mean_diff'] == ref['mean_diff'] and t['ci95'] == ref['ci_95']
            r, lo, hi = paired_ratio_ci(wa - wb, wb)
            assert t['ratio_to_cate_only'] == {'ratio': r, 'ci95': [lo, hi]}
        summaries[name] = s
    # the two variants differ in the population, hence in welfare
    w = {n: [c for c in s['cells'] if c['rule'] == 'tauV' and c['sigma_h'] == 0.0][0]['welfare']['mean']
         for n, s in summaries.items()}
    assert w['main'] != w['decoupled']


def test_simulate_rejects_unknown_dgp():
    import pytest
    with pytest.raises(ValueError):
        simulate(reps=1, n_players=500, dgp='nope')
    assert FEATURE_NAMES                                  # import sanity
