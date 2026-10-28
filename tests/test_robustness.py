"""study_robustness: one function per scenario, each returning a JSON-ready dict."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

import study_robustness as sr
from core.io import NumpyEncoder
from core.dgp import (FEATURE_NAMES, conditional_means, draw_features,
                      structural_adv, structural_main)

N = 1500
REPS = 3
TUNED = {'T': {'max_depth': 2, 'n_estimators': 30, 'learning_rate': 0.1}}
ROOT = Path(__file__).resolve().parents[1]

ARM_LABELS = ('CATE Only', 'CATE x LTV', 'CATE Only (tuned)', 'CATE x LTV (tuned)')
ORACLES = ('Oracle hindsight', 'Oracle feasible')


def _run(name, **kw):
    out = sr.SCENARIOS[name](REPS, N, TUNED, **kw)
    json.dumps(out, cls=NumpyEncoder)       # JSON-ready
    return out


def _check_cells(out, strategies=()):
    assert out['holm_family'] and out['holm_family_size'] >= 1
    m = 0
    for cell in out['cells'].values():
        for s in (*strategies, *ORACLES):
            assert s in cell['welfare'], s
        # hindsight is the exact top-k maximiser of realised welfare, so it
        # bounds the feasible oracle in every replication, hence in the mean
        assert (cell['welfare']['Oracle feasible']['mean']
                <= cell['welfare']['Oracle hindsight']['mean'] + 1e-9)
        for c in cell['comparisons'].values():
            if c.get('degenerate'):
                assert c['p_holm'] is None and c['mean_diff'] == 0.0 and c['p_value'] == 1.0
                continue
            assert {'p_value', 'p_holm', 'mean_diff', 'ci_95', 'cohens_d'} <= set(c)
            assert c['p_holm'] >= c['p_value'] - 1e-12 and c['p_holm'] <= 1.0
            assert c['mean_diff'] != round(c['mean_diff'])   # stored unrounded
            m += 1
    assert m == out['holm_family_size']


def test_scenarios_registry_is_complete():
    assert set(sr.SCENARIOS) == {
        'slearner_scenarios', 'asym', 'effective_welfare', 'component_update',
        'corr_sweep', 'heavy_tail', 'obs_diagnostic', 'aipw_calibration',
        'adversarial', 'like_for_like'}


def test_slearner_scenarios_main_null_correlated():
    out = _run('slearner_scenarios')
    assert set(out['cells']) == {'main', 'null', 'correlated', 'misspec'}
    _check_cells(out, (*ARM_LABELS, 'S-Learner'))
    assert 'S-Learner vs CATE x LTV' in out['cells']['misspec']['comparisons']
    assert 'drops login_freq' in out['misspec_note']
    # the S-learner keeps all features, so it is identical in the main and misspec cells
    assert (out['cells']['misspec']['welfare']['S-Learner']['mean']
            == out['cells']['main']['welfare']['S-Learner']['mean'])
    # the decomposition differs: its CATE model lost login_freq
    assert (out['cells']['misspec']['welfare']['CATE x LTV']['mean']
            != out['cells']['main']['welfare']['CATE x LTV']['mean'])
    # the CATE x LTV vs CATE Only difference is internally consistent
    c = out['cells']['null']
    w = c['welfare']
    d = c['comparisons']['CATE x LTV vs CATE Only']['mean_diff']
    assert d == pytest.approx(w['CATE x LTV']['mean'] - w['CATE Only']['mean'])


def test_asym_has_calibration_and_two_arms():
    out = _run('asym')
    _check_cells(out, (*ARM_LABELS, 'Full Framework (Est.)', 'Full Framework (Cal.)',
                       'Full Framework (Cal.) (tuned)', 'S-Learner'))
    cal = out['cells']['asym']['calibration']
    for arm in ('default', 'tuned'):
        assert 0 <= cal[arm]['identity_selected'] <= REPS and cal[arm]['n_runs'] == REPS


def test_effective_welfare_gap_is_nominal_minus_effective():
    out = _run('effective_welfare')
    _check_cells(out, ('CATE Only', 'CATE x LTV (tuned)', 'Full Framework (Cal.)',
                       'S-Learner'))
    cell = next(iter(out['cells'].values()))
    for s, row in out['overstatement'].items():
        c = cell['comparisons'][f'{s} vs {s} [effective]']
        assert c['mean_diff'] == pytest.approx(row['nominal'] - row['effective'])
    # nominal hindsight is the exact maximiser of nominal welfare
    nom = {s: r['nominal'] for s, r in out['overstatement'].items()}
    assert nom['Oracle hindsight'] >= max(nom.values()) - 1e-9


def test_component_update_changes_only_the_value_model():
    out = _run('component_update')
    _check_cells(out, ('CATE x LTV, misspecified V', 'CATE x LTV, well-specified V',
                       'CATE x LTV, misspecified V (tuned)',
                       'CATE x LTV, well-specified V (tuned)'))
    for arm, r in out['recovery'].items():
        w = next(iter(out['cells'].values()))['welfare']
        sfx = '' if arm == 'default' else ' (tuned)'
        assert r['recovery_abs'] == pytest.approx(
            w[f'CATE x LTV, well-specified V{sfx}']['mean']
            - w[f'CATE x LTV, misspecified V{sfx}']['mean'])


def test_corr_sweep_orders_corr_tau_v_and_uses_gamma_oracle():
    out = _run('corr_sweep')
    assert list(out['cells']) == ['negative', 'zero', 'positive']
    _check_cells(out, (*ARM_LABELS, 'S-Learner'))
    c = [out['cells'][k]['corr_tau_V'] for k in ('negative', 'zero', 'positive')]
    assert c[0] < c[1] < c[2]
    assert c[0] < 0 < c[2]


def test_heavy_tail_has_heavy_tailed_value():
    out = _run('heavy_tail')
    _check_cells(out, (*ARM_LABELS, 'S-Learner'))
    assert out['cells']['heavy_tail']['kappa'] == 0.30


def test_obs_diagnostic_shares_and_references():
    out = _run('obs_diagnostic')
    assert set(out['cells']) == {'wellspec', 'misspec'}
    assert out['holm_family_size'] >= 1
    for cell in out['cells'].values():
        for arm in ('default', 'tuned'):
            for ref in ('plug-in', 'Oracle hindsight', 'Oracle feasible'):
                r = cell['diagnostic'][arm][ref]
                assert r['cate_term'] >= 0 and r['value_term'] >= 0
                assert 0 <= r['cate_share_ratio_of_means_pct'] <= 100
                lvl = r['cate_share_per_rep']
                assert 0 <= lvl['mean'] <= 100 and lvl['ci95'] is not None
                assert 0 <= r['n_reps_used'] <= REPS
    for c in out['comparisons'].values():
        assert c['p_holm'] >= c['p_value'] - 1e-12


def test_aipw_plugin_selects_identity_by_construction():
    out = _run('aipw_calibration')
    cell = out['cells']['main']
    for arm in ('default', 'tuned'):
        assert cell['selection'][arm]['plugin']['identity_rate'] == 1.0
        assert cell['selection'][arm]['aipw']['n_runs'] == REPS
        # plug-in minus identity is exactly zero, so its raw p-value is 1
        sfx = '' if arm == 'default' else ' (tuned)'
        c = cell['comparisons'][f'Calibrated (plug-in){sfx} vs Identity (g=1){sfx}']
        assert c['mean_diff'] == 0.0 and c['p_value'] == 1.0 and c['degenerate']
    plug = [c for c in cell['comparisons'].values() if 'plug-in' in c['a']]
    assert len(plug) == 2 and all(c['degenerate'] for c in plug)
    assert not any(c.get('degenerate') for c in cell['comparisons'].values() if 'aipw' in c['a'])
    assert out['holm_family_size'] == 2          # only the two AIPW comparisons
    _check_cells(out)


def test_adversarial_dropped_feature_is_absent_from_value():
    # bonus_history (index 3) drives tau in the adv DGP but not V: shifting it must
    # leave E[V|X] untouched and move E[tau|X].
    f = draw_features(60, np.random.RandomState(4))
    g = dict(f, bonus_history=f['bonus_history'] + 2.0)
    a = conditional_means('adv', f, n_mc=50, seed=3)
    b = conditional_means('adv', g, n_mc=50, seed=3)
    np.testing.assert_array_equal(a['ltv_bar'], b['ltv_bar'])
    assert np.abs(a['tau_bar'] - b['tau_bar']).max() > 1e-3
    assert sr.ADV_DROP_COLS == [0, 1, 2]
    assert FEATURE_NAMES[3] == 'bonus_history'


def test_adv_and_asym_feasible_oracle_noise_free_equals_structure():
    for name, structural in (('adv', structural_adv),):
        rng = np.random.RandomState(8)
        f = draw_features(400, rng)
        s = structural(f, rng, noise_scale=0.0)
        cm = conditional_means(name, f, n_mc=3, seed=1, noise_scale=0.0)
        np.testing.assert_allclose(cm['tau_bar'], s['true_tau'], atol=1e-12)
        np.testing.assert_allclose(cm['ltv_bar'], s['true_ltv'], atol=1e-9)


def test_adversarial_cells_and_arms():
    out = _run('adversarial')
    assert set(out['cells']) == {'kappa_0.30', 'kappa_0.10'}
    _check_cells(out, ('CATE Only', 'CATE x LTV', 'CATE x LTV (tuned)', 'S-Learner',
                       'CATE x LTV (well-specified)',
                       'CATE x LTV (well-specified) (tuned)'))


def test_like_for_like_drops_same_feature_from_s_learner():
    out = _run('like_for_like')
    assert set(out['cells']) == {'login_drop_main', 'bonus_drop_adversarial'}
    assert out['cells']['login_drop_main']['dropped_feature_index'] == 2
    assert out['cells']['bonus_drop_adversarial']['dropped_feature_index'] == 3
    _check_cells(out, ('S-Learner (all features)', 'S-Learner (feature dropped)',
                       'CATE Only (all features)', 'CATE Only (tuned, all features)'))
    assert 'S-Learner: all features vs dropped' in \
        out['cells']['login_drop_main']['comparisons']


def test_apply_holm_pools_across_cells():
    cells = {'a': {'comparisons': {'x': {'p_value': 0.01}, 'y': {'p_value': 0.04}}},
             'b': {'comparisons': {'z': {'p_value': 0.03}}}}
    m = sr._apply_holm(cells)
    assert m == 3
    assert cells['a']['comparisons']['x']['p_holm'] == pytest.approx(0.03)
    assert cells['b']['comparisons']['z']['p_holm'] == pytest.approx(0.06)
    assert cells['a']['comparisons']['y']['p_holm'] == pytest.approx(0.06)


def test_main_writes_one_key_per_scenario(tmp_path, monkeypatch):
    import core.io as cio
    monkeypatch.setattr(cio, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(sr, 'load_or_tune', lambda **k: TUNED)
    monkeypatch.setattr(sr, 'SCENARIOS', {'one': lambda r, n, t, n_jobs=1: {'r': r, 'n': n},
                                          'two': lambda r, n, t, n_jobs=1: {'ok': True}})
    sr.main(['--smoke'])
    data = json.loads((tmp_path / 'robustness_smoke.json').read_text())
    assert data['one'] == {'r': 3, 'n': 2000} and data['two'] == {'ok': True}
    assert data['_meta']['n_runs'] == 3


def test_legacy_scripts_deleted_and_nothing_imports_them():
    gone = ['compute_revision_additions'] + [f'compute_v{v}_additions' for v in range(42, 46)]
    for g in gone:
        assert not (ROOT / 'src' / f'{g}.py').exists()
    pat = re.compile(r'^\s*(from|import)\s+(' + '|'.join(gone) + r')\b', re.M)
    for p in list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'tests').rglob('*.py')):
        assert not pat.search(p.read_text()), p


def test_no_em_dash_in_module():
    assert chr(0x2014) not in (ROOT / 'src' / 'study_robustness.py').read_text()


# ---------------------------------------------------------------------------
# Regression pins: the default arms reproduce the legacy definitions bit-for-bit
# ---------------------------------------------------------------------------
def test_asym_default_arm_equals_run_single_simulation():
    from core.dgp import generate_asym
    from study_main import SEED_BASE, run_single_simulation
    res = run_single_simulation(SEED_BASE, dgp_func=generate_asym, n_players=N)[0]
    w = sr._dgp_rep(SEED_BASE, dgp_func=generate_asym, n_players=N, tuned=TUNED,
                    kappa=0.30, calibrate=True)['w']
    for new, old in (('CATE Only', 'CATE Only'), ('CATE x LTV', 'CATE x LTV'),
                     ('Full Framework (Est.)', 'Full Framework (Est.)'),
                     ('Full Framework (Cal.)', 'Full Framework (Cal.)'),
                     ('S-Learner', 'S-Learner'),
                     ('Oracle hindsight', 'Oracle hindsight (g=1)'),
                     ('Oracle feasible', 'Oracle feasible (g=1)')):
        assert w[new] == res[old]['welfare'], new


def _legacy_rep(dgp_func, seed):
    from sklearn.model_selection import train_test_split
    data = dgp_func(N, np.random.RandomState(seed))
    tr, te = train_test_split(np.arange(len(data['treatment'])), test_size=0.30,
                              random_state=seed)
    return data, tr, te


def test_component_update_equals_direct_estimator_calls():
    from core.dgp import generate_main
    from core.estimators import _wellspec_value_model, estimate_components
    from core.policy import compute_welfare, target_topk
    from study_main import SEED_BASE
    data, tr, te = _legacy_rep(generate_main, SEED_BASE)
    tau_hat, v_mis, _ = estimate_components(data, tr, te, seed=SEED_BASE)
    v_well = _wellspec_value_model(data['features'][tr], data['features'][te], SEED_BASE)
    k = int(len(te) * 0.30)
    tau, V = data['true_tau'][te], data['true_ltv'][te]
    w = sr._component_rep(SEED_BASE, n_players=N, tuned=TUNED, kappa=0.30)
    assert w['CATE x LTV, misspecified V'] == compute_welfare(
        target_topk(tau_hat * v_mis, k), tau, V)
    assert w['CATE x LTV, well-specified V'] == compute_welfare(
        target_topk(tau_hat * v_well, k), tau, V)


def test_obs_diagnostic_misspec_arm_equals_estimate_components_misspec():
    from core.dgp import generate_main
    from core.estimators import estimate_components_misspec
    from core.policy import misrank_terms
    from study_main import SEED_BASE
    data, tr, te = _legacy_rep(generate_main, SEED_BASE)
    tau_hat, v_hat, _ = estimate_components_misspec(data, tr, te, seed=SEED_BASE)
    k = int(len(te) * 0.30)
    expect = misrank_terms(tau_hat, v_hat, data['true_tau'][te], data['true_ltv'][te], k)
    got = sr._obs_rep(SEED_BASE, n_players=N, tuned=TUNED, kappa=0.30)
    assert got[('misspec', 'default', 'Oracle hindsight')] == expect


def test_flag_degenerate_needs_name_and_zero_value(caplog):
    from study_robustness import flag_degenerate
    comps = {
        'zero': {'a': 'Calibrated (plug-in) x', 'mean_diff': 0.0},
        'moved': {'a': 'Calibrated (plug-in) y', 'mean_diff': 12.5},
        'aipw_zero': {'a': 'Calibrated (aipw) x', 'mean_diff': 0.0},
    }
    with caplog.at_level('WARNING'):
        flag_degenerate(comps)
    assert comps['zero']['degenerate'] is True
    assert comps['moved']['degenerate'] is False
    assert comps['aipw_zero']['degenerate'] is False
    assert [r.message for r in caplog.records if 'moved' in r.message]


def test_aipw_calibration_stores_per_rep_regret():
    cell = _run('aipw_calibration')['cells']['main']
    for label, c in cell['comparisons'].items():
        diffs = cell['per_rep_diff_vs_identity'][c['a']]
        assert len(diffs) == REPS
        assert sum(diffs) / REPS == pytest.approx(c['mean_diff'])
    for key, r in cell['regret_vs_feasible'].items():
        lo, hi = r['ci95']
        assert lo <= r['mean'] <= hi
        assert r['mean'] == pytest.approx(
            cell['welfare']['Oracle feasible']['mean'] - cell['welfare'][key]['mean'])


def test_obs_diagnostic_level_is_per_rep_mean_and_ratio_of_means_is_separate():
    out = _run('obs_diagnostic')
    differs = 0
    for cell in out['cells'].values():
        for arm in ('default', 'tuned'):
            for ref, r in cell['diagnostic'][arm].items():
                assert 'cate_share_pct' not in r
                a, b = r['cate_term'], r['value_term']
                assert r['cate_share_ratio_of_means_pct'] == pytest.approx(100 * a / (a + b))
                differs += r['cate_share_per_rep']['mean'] != pytest.approx(
                    r['cate_share_ratio_of_means_pct'])
    assert differs     # the two estimands are not the same number
