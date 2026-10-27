"""Tuning protocol (frozen, deterministic, disjoint pilot seeds) and the study's wiring."""
import numpy as np

import study_learners as sl
from simulation_v38 import N_RUNS, SEED_BASE

TINY_GRID = [{'max_depth': 2, 'n_estimators': 20, 'learning_rate': 0.1},
             {'max_depth': 3, 'n_estimators': 40, 'learning_rate': 0.1}]


def test_pilot_seeds_disjoint_from_evaluation_seeds():
    eval_seeds = {SEED_BASE + 7 * r for r in range(N_RUNS)}
    pilot = sl.pilot_seeds()
    assert pilot == [10_000 + 7 * r for r in range(5)]
    assert not set(pilot) & eval_seeds


def test_grid_is_the_twelve_configs():
    assert len(sl.GRID) == 12
    assert {(g['max_depth'], g['n_estimators'], g['learning_rate']) for g in sl.GRID} == {
        (d, n, lr) for d in (2, 3, 4) for n in (100, 300) for lr in (0.05, 0.1)}


def test_tuning_selection_deterministic():
    kw = dict(grid=TINY_GRID, seeds=sl.pilot_seeds(2), n_players=1200)
    a = sl.tune_learner('DR', **kw)
    b = sl.tune_learner('DR', **kw)
    assert a == b
    assert a['params'] in TINY_GRID
    assert a['mean_loss'] == min(a['losses'])


def test_load_or_tune_reads_frozen_file_without_pilot(tmp_path, monkeypatch):
    monkeypatch.setattr(sl, 'OUTPUT_DIR', tmp_path)
    frozen = {'learners': {l: {'params': sl.DEFAULT_PARAMS[l]} for l in sl.LEARNERS}}
    (tmp_path / sl.TUNED_FILE).write_text(__import__('json').dumps(frozen))
    monkeypatch.setattr(sl, 'tune_all', lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert sl.load_or_tune()['T'] == sl.DEFAULT_PARAMS['T']


def test_simulate_shapes_holm_and_regret():
    tuned = {l: TINY_GRID[0] for l in sl.LEARNERS}
    raw = sl.simulate(3, 1500, tuned, learners=('T', 'S'), kappas=(0.1, 0.3))
    assert raw['welfare'][sl.E2E_KEY].shape == (3, 2)
    assert raw['welfare']['T (default)|misspec|tauV'].shape == (3, 2)
    out = sl.summarise(raw)
    # 2 learners x 2 psets x 2 vms x 2 rules - baseline + E2E
    assert out['n_comparisons_per_kappa'] == 2 * 2 * 2 * 2 - 1 + 1
    for rows in out['tests_vs_T_default'].values():
        assert all(r['p_holm'] >= r['p_raw'] for r in rows)
    r = out['regret_split_kappa_0.30']['T (tuned)|misspec']
    assert abs(r['shapley_tau_share_of_mean'] + r['shapley_v_share_of_mean'] - 1) < 1e-9
