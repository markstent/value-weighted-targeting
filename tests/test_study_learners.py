"""Tuning protocol (frozen, deterministic, disjoint pilot seeds) and the study's wiring."""
import numpy as np
import pytest

import study_learners as sl
from study_main import N_RUNS, SEED_BASE

TINY_GRID = [{'max_depth': 2, 'n_estimators': 20, 'learning_rate': 0.1},
             {'max_depth': 3, 'n_estimators': 40, 'learning_rate': 0.1}]


def test_pilot_seeds_disjoint_from_evaluation_seeds():
    eval_seeds = {SEED_BASE + 7 * r for r in range(N_RUNS)}
    pilot = sl.pilot_seeds()
    assert pilot == [10_000 + 7 * r for r in range(5)]
    assert not set(pilot) & eval_seeds


def test_grid_is_the_eighty_configs():
    assert len(sl.GRID) == 80
    assert {(g['max_depth'], g['n_estimators'], g['learning_rate']) for g in sl.GRID} == {
        (d, n, lr) for d in (1, 2, 3, 4) for n in (25, 50, 100, 300)
        for lr in (0.005, 0.01, 0.02, 0.05, 0.1)}


def test_r_not_in_comparison_and_boundary_check():
    assert 'R' not in sl.COMPARED_LEARNERS
    assert sl.boundary_check(sl.GRID, {'max_depth': 1, 'n_estimators': 25,
                                       'learning_rate': 0.005}) == {
        'max_depth': 'hard_floor', 'n_estimators': 'floor', 'learning_rate': 'floor'}
    assert sl.boundary_check(sl.GRID, {'max_depth': 4, 'n_estimators': 100,
                                       'learning_rate': 0.05}) == {
        'max_depth': 'ceiling', 'n_estimators': 'interior', 'learning_rate': 'interior'}


def test_tuning_selection_deterministic():
    kw = dict(grid=TINY_GRID, seeds=sl.pilot_seeds(2), n_players=1200)
    a = sl.tune_learner('DR', **kw)
    b = sl.tune_learner('DR', **kw)
    assert a == b
    assert a['params'] in TINY_GRID
    assert a['mean_loss'] == min(a['losses'])


def _write_frozen(tmp_path, monkeypatch, n=sl.N_PLAYERS, **meta_over):
    import json
    monkeypatch.setattr(sl, 'CONFIG_DIR', tmp_path)
    monkeypatch.setattr(sl, 'OUTPUT_DIR', tmp_path)
    frozen = {'learners': {lrn: {'params': sl.DEFAULT_PARAMS[lrn]} for lrn in sl.COMPARED_LEARNERS},
              '_meta': {**sl.make_meta(n), **meta_over}}
    path = sl.tuned_path(n)
    path.write_text(json.dumps(frozen))
    monkeypatch.setattr(sl, 'tune_all', lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    return path


def test_load_or_tune_reads_frozen_file_without_pilot(tmp_path, monkeypatch):
    _write_frozen(tmp_path, monkeypatch)
    assert sl.load_or_tune()['T'] == sl.DEFAULT_PARAMS['T']


def test_load_or_tune_meta_mismatch_raises_with_hint(tmp_path, monkeypatch):
    for key, bad in (('pilot_seeds', [1, 2, 3, 4, 5]), ('n_players', 123),
                     ('dgp', 'null'), ('grid', sl.GRID[:3])):
        _write_frozen(tmp_path, monkeypatch, **{key: bad})
        with pytest.raises(ValueError, match=r"--retune"):
            sl.load_or_tune()


def test_small_runs_never_use_tracked_config(tmp_path, monkeypatch):
    monkeypatch.setattr(sl, 'CONFIG_DIR', tmp_path / 'config')
    monkeypatch.setattr(sl, 'OUTPUT_DIR', tmp_path / 'out')
    assert sl.tuned_path(sl.N_PLAYERS).parent == tmp_path / 'config'
    assert sl.tuned_path(2000).parent == tmp_path / 'out'
    monkeypatch.setattr(sl, 'tune_all', lambda n, j: {
        'learners': {lrn: {'params': sl.DEFAULT_PARAMS[lrn]} for lrn in sl.COMPARED_LEARNERS},
        '_meta': sl.make_meta(n)})
    sl.load_or_tune(n_players=2000, retune=True)
    assert (tmp_path / 'out' / 'tuned_params_n2000.json').exists()
    assert not (tmp_path / 'config').exists()


def test_simulate_shapes_holm_and_regret():
    tuned = {lrn: TINY_GRID[0] for lrn in sl.COMPARED_LEARNERS}
    raw = sl.simulate(3, 1500, tuned, learners=('T', 'S'), kappas=(0.1, 0.3))
    assert raw['welfare'][sl.E2E_KEY].shape == (3, 2)
    assert raw['welfare']['T (default)|misspec'].shape == (3, 2)
    assert raw['welfare']['T (default)|tau'].shape == (3, 2)
    assert not any(k.endswith('|tauV') or k.count('|') > 1 for k in raw['welfare'])
    out = sl.summarise(raw)
    # tauV: 4 combos x 2 vms - 2 baselines; tau only: 4 combos - 1 baseline; plus E2E;
    # plus S (tuned) vs T (tuned) for the three value models (Holm family 1); effect-only vs tauV
    # for every learner x tuning x value model
    # (2 x 2 x 2) is its own Holm family 2, never pooled with family 1
    assert out['n_comparisons_per_kappa'] == 6 + 3 + 1 + 3
    assert out['n_effect_only_comparisons_per_kappa'] == 8
    for kp, rows in out['tests_S_vs_T_tuned'].items():
        assert [(r['combo'], r['baseline']) for r in rows] == [
            (f'S (tuned)|{vm}', f'T (tuned)|{vm}') for vm in ('misspec', 'wellspec', 'tau')]
        for r in rows:
            assert r['ci95'][0] <= r['diff'] <= r['ci95'][1]
            assert r['p_holm'] >= r['p_raw']
        assert len(out['tests_vs_T_default'][kp]) + len(rows) == out['n_comparisons_per_kappa']
    assert not any(r['combo'] == r['baseline'] for rows in out['tests_vs_T_default'].values()
                   for r in rows)
    for rows in out['tests_vs_T_default'].values():
        assert all(r['p_holm'] >= r['p_raw'] for r in rows)
    r = out['regret_split_kappa_0.30']['T (tuned)|misspec']
    assert abs(r['shapley_tau_share_of_mean'] + r['shapley_v_share_of_mean'] - 1) < 1e-9


def test_effect_only_vs_tauv_tests_keys_and_invariants():
    """G8: effect-only minus tau_hat*V_hat, per learner x tuning x value model and kappa; its
    own Holm family per kappa, with the pairing pinned to paired_test_raw on the stored arrays."""
    from core.stats import holm_bonferroni, paired_ratio_ci, paired_test_raw
    tuned = {lrn: TINY_GRID[0] for lrn in sl.COMPARED_LEARNERS}
    raw = sl.simulate(3, 1500, tuned, learners=('T', 'S'), kappas=(0.1, 0.5))
    out = sl.summarise(raw)
    assert set(out['tests_effect_only_vs_tauV']) == {'0.1', '0.5'}
    for kp, rows in out['tests_effect_only_vs_tauV'].items():
        assert len(rows) == 8
        assert {(r['combo'], r['baseline']) for r in rows} == {
            (f'{lrn} ({p})|tau', f'{lrn} ({p})|{vm}')
            for lrn in ('T', 'S') for p in ('tuned', 'default') for vm in ('misspec', 'wellspec')}
        j = out['kappas'].index(float(kp))
        for r in rows:
            wa, wb = raw['welfare'][r['combo']][:, j], raw['welfare'][r['baseline']][:, j]
            ref = paired_test_raw(wa, wb)
            assert r['diff'] == ref['mean_diff'] and r['se_diff'] == ref['se_diff']
            assert np.isclose(r['t_stat'], ref['t_stat'], equal_nan=True) and r['p_raw'] == ref['p_value']
            assert r['ratio_to_baseline']['ratio'] == paired_ratio_ci(wa - wb, wb)[0]
            assert r['family_size'] == 8
        # Holm over exactly the effect-only family, not pooled with the learner family
        assert [r['p_holm'] for r in rows] == list(holm_bonferroni([r['p_raw'] for r in rows]))
        fam1 = out['tests_vs_T_default'][kp] + out['tests_S_vs_T_tuned'][kp]
        assert [r['p_holm'] for r in fam1] == list(holm_bonferroni([r['p_raw'] for r in fam1]))


def test_check_meta_compares_learners_and_criterion():
    expected = sl.make_meta(sl.N_PLAYERS)
    sl._check_meta(dict(expected), expected, 'p')                        # identical: fine
    for key, bad in (('learners', ['T', 'X']), ('criterion', 'something else')):
        with pytest.raises(ValueError, match=key):
            sl._check_meta(dict(expected, **{key: bad}), expected, 'p')
