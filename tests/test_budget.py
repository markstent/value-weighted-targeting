"""kappa* 2x2 diagnostic: oracle/estimated tau x oracle/misspec/wellspec V."""
import numpy as np

import study_budget as sb


def test_oracle_both_no_crossover():
    raw = sb.simulate(reps=5, n_players=2000, tau_sources=('oracle',), v_sources=('oracle',))
    cell = raw['oracle|oracle']
    diff = cell['tauV'] - cell['tau']
    assert diff.shape == (5, len(sb.budget_grid))
    assert np.all(cell['tauV'] >= cell['tau'])
    # Mean difference never changes sign, so no kappa* exists.
    assert np.isnan(sb.crossover_kappa(diff.mean(axis=0), sb.budget_grid))


def test_frac_reported():
    raw = sb.simulate(reps=3, n_players=1500, tau_sources=('oracle', 'T'),
                      v_sources=('oracle', 'misspec'))
    out = sb.summarise(raw, n_boot=200)
    assert set(out['cells']) == {'oracle|oracle', 'oracle|misspec', 'T|oracle', 'T|misspec'}
    for cell in out['cells'].values():
        f = cell['kappa_star']['frac_with_crossover']
        assert 0.0 <= f <= 1.0
        assert len(cell['per_kappa']) == len(sb.budget_grid)


def test_frac_counts_no_crossover_resamples():
    # Two reps: A crosses (-1 -> +3), B never does (-1 -> -1).  A resample's mean
    # curve crosses iff it contains A at least once: probability 3/4.
    diff = np.array([[-1.0, 3.0], [-1.0, -1.0]])
    grid = np.array([0.1, 0.2])
    res = sb.bootstrap_crossover(diff, grid, n_boot=2000, seed=1)
    assert res['n_boot'] == 2000
    assert res['n_crossing'] + res['n_below_grid'] + res['n_above_grid'] == 2000
    assert res['n_no_crossover'] > 0
    assert res['frac_with_crossover'] == res['n_crossing'] / 2000
    assert abs(res['frac_with_crossover'] - 0.75) < 0.04
    # Replay the same resampling independently.
    rng = np.random.RandomState(1)
    idx = rng.randint(0, 2, (2000, 2))
    expected = int(np.sum((idx == 0).any(axis=1)))
    assert res['n_crossing'] == expected
    assert res['n_above_grid'] == 2000 - expected and res['n_below_grid'] == 0


def test_no_crossover_anywhere_gives_zero_frac_and_no_ci():
    diff = np.full((4, 3), -1.0)
    res = sb.bootstrap_crossover(diff, np.array([0.1, 0.2, 0.3]), n_boot=100, seed=0)
    assert res['frac_with_crossover'] == 0.0 and res['n_above_grid'] == 100
    assert res['ci95_conditional_on_crossing'] is None
    assert res['ci95'] == [0.3, 0.3] and res['ci95_labels'] == ['>max', '>max']


def test_holm_family_is_all_kappa():
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, (30, len(sb.budget_grid)))
    b = rng.normal(0, 1, (30, len(sb.budget_grid)))
    rows = sb.paired_inference(a, b, sb.budget_grid)
    assert len(rows) == len(sb.budget_grid)
    raw = np.array([r['p_raw'] for r in rows])
    holm = np.array([r['p_holm'] for r in rows])
    assert np.all(holm >= raw - 1e-12)
    # The smallest raw p is multiplied by the family size (all kappa), not a subset.
    assert np.isclose(holm.min(), min(1.0, raw.min() * len(raw)))


G = np.array([0.1, 0.2, 0.3, 0.4, 0.5])


def test_kappa_star_semantics():
    cls = sb.classify_kappa_star
    # down-only: tau only leads at the top of the grid
    k, o, n = cls([1.0, 0.5, 0.0, -0.5, -1.0], G)
    assert np.isnan(k) and o == 'above_grid' and n == 1
    # several crossings, last stable up-crossing wins
    k, o, n = cls([-1.0, 1.0, -1.0, 1.0, 2.0], G)
    assert (k, o, n) == (0.4, 'crossing', 3)
    # starts positive and stays positive
    k, o, n = cls([1.0, 2.0, 3.0, 4.0, 5.0], G)
    assert np.isnan(k) and o == 'below_grid' and n == 0
    # a tie at 0 is not > 0
    k, o, _ = cls([-1.0, 0.0, 1.0, 1.0, 1.0], G)
    assert (k, o) == (0.3, 'crossing')
    k, o, _ = cls([-1.0, 0.0, 0.0, 0.0, 0.0], G)
    assert np.isnan(k) and o == 'above_grid'


def test_unconditional_ci_labels_censored_bounds():
    diff = np.tile([1.0, 1.0, 1.0], (4, 1))
    res = sb.bootstrap_crossover(diff, np.array([0.1, 0.2, 0.3]), n_boot=50, seed=0)
    assert res['n_below_grid'] == 50
    assert res['ci95'] == [0.1, 0.1] and res['ci95_labels'] == ['<min', '<min']


def test_paired_test_single_implementation():
    from core.stats import paired_test, paired_test_raw
    rng = np.random.RandomState(3)
    a, b = rng.normal(100, 20, 40), rng.normal(95, 20, 40)
    raw, rnd = paired_test_raw(a, b), paired_test(a, b)
    assert rnd['mean_diff'] == round(raw['mean_diff'], 0)
    assert rnd['p_value'] == round(raw['p_value'], 4)
    assert rnd['cohens_d'] == round(raw['cohens_d'], 3)
    row = sb.paired_inference(a[:, None], b[:, None], [0.1])[0]
    assert row['p_raw'] == raw['p_value'] and row['cohens_d'] == raw['cohens_d']
    assert row['se_diff'] == raw['se_diff']
