"""Harm module: correlation sweep, estimated harm, solver feasibility."""
import numpy as np

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


def test_hard_cap_binds_at_sigma0():
    raw = simulate(reps=3, n_players=3000, c_harms=[1.0], sigma_hs=[0.0], kappas=[0.30])
    for rule in ('hard cap', 'hard cap (oracle tau,V)'):
        cell = raw[(1.0, 0.0, 0.30, rule)]
        gap = np.array(cell['cap']) - np.array(cell['exposure'])
        assert np.all(np.abs(gap) < 0.005), gap


def test_selection_never_sees_true_harm():
    import inspect

    from study_harm import select_rules
    assert not any('true' in p for p in inspect.signature(select_rules).parameters)

    kw = dict(reps=2, n_players=2000, c_harms=[1.0], sigma_hs=[0.0, 0.5], kappas=[0.30])
    perm = np.random.RandomState(11).permutation(600)    # test split has 600 rows
    rec_a, rec_b = {}, {}
    raw_a = simulate(**kw, _record=rec_a)
    # true harm permuted AFTER harm_hat is computed and frozen
    raw_b = simulate(**kw, _record=rec_b, _eval_harm_transform=lambda h: h[perm])
    assert rec_a and set(rec_a) == set(rec_b)
    for key in rec_a:
        np.testing.assert_array_equal(rec_a[key], rec_b[key], err_msg=str(key))
    # welfare uses true tau, V only: identical.  Harm metrics must differ.
    for key in raw_a:
        assert raw_a[key]['welfare'] == raw_b[key]['welfare']
    assert any(raw_a[k]['exposure'] != raw_b[k]['exposure'] for k in raw_a)


def test_soft_gate_has_no_violation_rate_and_oracle_flag():
    raw = simulate(reps=2, n_players=2000, c_harms=[1.0], sigma_hs=[0.0, 0.5], kappas=[0.30])
    s = summarise(raw, n_boot=50)
    for e in s['cells']:
        assert e['oracle_harm'] == (e['sigma_h'] == 0.0)
        if e['rule'].startswith('soft gate'):
            assert e['cap_violation_rate'] is None and e['overshoot'] is not None
        if e['rule'].startswith('hard cap'):
            assert e['cap_violation_rate'] is not None and e['overshoot'] is not None
        assert 0.3 < e['pop_mean_harm'] < 0.6 and 0 < e['share_harm_gt_0_6'] < 1


def test_paired_tests_keys_and_holm_family():
    """G1/G2: per-rep paired tests in every cell, estimated and feasible-oracle components,
    one Holm family (the cell's eight tests: six + soft gate vs CATE only, both sources); statistics pinned to
    paired_test_raw on the raw
    per-rep welfare arrays, p_holm to holm_bonferroni over exactly that family."""
    from core.stats import holm_bonferroni, paired_ratio_ci, paired_test_raw
    raw = simulate(reps=4, n_players=3000, c_harms=[1.0], sigma_hs=[0.0, 0.5], kappas=[0.30])
    summary = summarise(raw, n_boot=50)
    tests = summary['paired_tests']
    assert len(tests) == 2 * 8
    assert {t['comparison'] for t in tests} == {'hard cap vs soft gate', 'soft gate vs tauV',
                                                 'hard cap vs tauV',
                                                 'soft gate vs CATE only'}
    assert {t['components'] for t in tests} == {'estimated', 'oracle'}
    for t in tests:
        assert t['dgp'] == 'main'
        key = (t['c_harm'], t['sigma_h'], t['kappa'])
        wa = np.array(raw[key + (t['a'],)]['welfare'])
        wb = np.array(raw[key + (t['b'],)]['welfare'])
        ref = paired_test_raw(wa, wb)
        assert t['mean_diff'] == ref['mean_diff'] and t['se_diff'] == ref['se_diff']
        assert np.isclose(t['t_stat'], ref['t_stat'], equal_nan=True) and t['p_raw'] == ref['p_value']
        assert t['ci95'] == ref['ci_95'] and t['cohens_d'] == ref['cohens_d']
        blind = np.array(raw[key + ('tauV' + ('' if t['components'] == 'estimated'
                                              else ' (oracle tau,V)'),)]['welfare'])
        assert t['ratio_to_blind']['ratio'] == paired_ratio_ci(wa - wb, blind)[0]
        assert t['family_size'] == 8 and t['p_holm'] >= t['p_raw']
    for sigma in (0.0, 0.5):                      # Holm over exactly the cell's eight tests
        fam = [t for t in tests if t['sigma_h'] == sigma]
        assert len(fam) == 8
        assert [t['p_holm'] for t in fam] == list(holm_bonferroni([t['p_raw'] for t in fam]))
    oracle = [t for t in tests if t['components'] == 'oracle']
    assert all('(oracle tau,V)' in t['a'] and '(oracle tau,V)' in t['b'] for t in oracle)
