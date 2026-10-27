"""Budget crossover kappa*: which component drives it?

Below kappa*, ranking on tau alone ("tau only") beats ranking on tau * V
("tauV"); above it the order reverses.  With TRUE components tau * V dominates at
every kappa, so kappa* is an estimation phenomenon.  This study isolates the
source with a 4 x 3 grid of cells, each a pair (tau source, V source):

  tau source:  oracle (feasible E[tau|X]), T, X or S (that CATE learner's tau_hat)
  V source:    oracle (feasible E[V|X]), misspec (the deliberately misspecified
               proxy), wellspec (the well-specified value model)

For each cell, the score tau_src * V_src is compared with tau_src alone at every
kappa in budget_grid.  Welfare is ALWAYS evaluated on the realised true tau and V
of the test split.  Every learner and value model is fitted ONCE per replication
on the training split; conditional_means is called once per replication on the
test features; all cells and kappas reuse them.

S-learner: only its tau_hat = mu(x,1) - mu(x,0) is used (never a tau*V joint
output).  That tau_hat does not take ltv_hat as an input at all, so it is the
same in every cell; the misspecified proxy's ltv_hat is only ever a V source.

kappa*: the smallest grid kappa at which the mean welfare difference (tauV minus
tau only) is > 0 there and at every larger grid kappa (a stable up-crossing; a tie
at 0 is not > 0).  It is a grid value, not an interpolation.  Every point estimate
and bootstrap resample is classified: "crossing" (kappa* above the grid minimum),
"below_grid" (diff > 0 everywhere, tauV leads throughout) or "above_grid" (diff <=
0 at the grid maximum, tau only still leads).  All three counts are reported, plus
n_sign_changes across the grid.  The CI is an UNCONDITIONAL percentile bootstrap
over replications (2000 resamples, seeded via seed_from): below_grid sorts as the
grid minimum, above_grid as the grid maximum, and a bound that is such a censored
value is labelled "<min" / ">max".  ci95_conditional_on_crossing uses crossing
resamples only.  frac_with_crossover = n_crossing / n_boot.

Inference: per cell, core.stats.paired_test_raw of tauV vs tau only at each kappa,
Holm (core.stats.holm_bonferroni) across ALL kappa of that cell (the family is every
kappa for one comparison).  Everything is unrounded in the JSON.

Usage:  python study_budget.py [--reps R] [--n-players N] [--smoke]
"""

import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from simulation_v38 import N_PLAYERS, N_RUNS, SEED_BASE
from core.dgp import DGPS, FEATURE_NAMES, conditional_means
from core.estimators import (
    BOOT_STREAM, ORACLE_STREAM, _estimate_cate_slearner, _estimate_cate_tlearner,
    _estimate_cate_xlearner, _estimate_ltv_proxy, _wellspec_value_model, seed_from,
)
from core.io import add_common_args, resolve_run_size, save_json
from core.policy import compute_welfare, target_topk
from core.stats import holm_bonferroni, paired_test_raw, t_mean_ci as _t_ci

# Rounded so int(n * kappa) is identical to the grid the published v44 numbers used.
budget_grid = np.round(np.arange(0.05, 0.501, 0.025), 3)
REPORTED_BUDGETS = (0.10, 0.20, 0.30, 0.50)
TAU_SOURCES = ('oracle', 'T', 'X', 'S')
V_SOURCES = ('oracle', 'misspec', 'wellspec')
ORACLE_MC = 200
N_BOOT = 2000


def welfare_at_kappa(scores, kappa, true_tau, true_ltv):
    """True welfare of the top-k set by `scores`, k = int(n * kappa)."""
    k = int(len(scores) * kappa)
    return compute_welfare(target_topk(scores, k), true_tau, true_ltv)


def cell_key(tau_src, v_src):
    return f'{tau_src}|{v_src}'


def _tau_hat(source, Xtr, Xte, Wtr, Ytr):
    return {'T': _estimate_cate_tlearner, 'X': _estimate_cate_xlearner,
            'S': _estimate_cate_slearner}[source](Xtr, Xte, Wtr, Ytr)


def simulate(reps, n_players, dgp_name='main', tau_sources=TAU_SOURCES,
             v_sources=V_SOURCES, kappas=budget_grid):
    """Per-rep welfare of tauV and tau-only for every (tau source, V source) cell.

    Returns {cell_key: {'tauV': (reps, K) array, 'tau': (reps, K) array}}."""
    kappas = np.asarray(kappas, dtype=float)
    keys = [cell_key(t, v) for t in tau_sources for v in v_sources]
    out = {key: {'tauV': np.empty((reps, len(kappas))), 'tau': np.empty((reps, len(kappas)))}
           for key in keys}
    for rep in range(reps):
        seed = SEED_BASE + rep * 7
        data = DGPS[dgp_name](n_players, np.random.RandomState(seed))
        idx = np.arange(len(data['treatment']))
        train_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=seed)
        true_tau = data['true_tau'][test_idx]
        true_ltv = data['true_ltv'][test_idx]
        X, W, Y = data['features'], data['treatment'], data['y_obs']
        Xtr, Xte, Wtr, Ytr = X[train_idx], X[test_idx], W[train_idx], Y[train_idx]

        # One fit per component per rep, reused across cells and kappas.
        taus, vs = {}, {}
        for src in tau_sources:
            if src == 'oracle':
                continue
            taus[src] = _tau_hat(src, Xtr, Xte, Wtr, Ytr)
        if 'misspec' in v_sources:
            vs['misspec'] = _estimate_ltv_proxy(Xtr, Xte, seed)
        if 'wellspec' in v_sources:
            vs['wellspec'] = _wellspec_value_model(Xtr, Xte, seed)
        if 'oracle' in tau_sources or 'oracle' in v_sources:
            feats = {name: X[test_idx, i] for i, name in enumerate(FEATURE_NAMES)}
            cm = conditional_means(dgp_name, feats, n_mc=ORACLE_MC,
                                   seed=seed_from(seed, ORACLE_STREAM))
            taus['oracle'], vs['oracle'] = cm['tau_bar'], cm['ltv_bar']

        for t in tau_sources:
            for v in v_sources:
                cell = out[cell_key(t, v)]
                s_tauv, s_tau = taus[t] * vs[v], taus[t]
                for j, kappa in enumerate(kappas):
                    cell['tauV'][rep, j] = welfare_at_kappa(s_tauv, kappa, true_tau, true_ltv)
                    cell['tau'][rep, j] = welfare_at_kappa(s_tau, kappa, true_tau, true_ltv)
    return out


CROSSING, BELOW_GRID, ABOVE_GRID = 'crossing', 'below_grid', 'above_grid'
_OUTCOME_RANK = {BELOW_GRID: 0, CROSSING: 1, ABOVE_GRID: 2}


def classify_kappa_star(mean_diff, grid):
    """(kappa*, outcome, n_sign_changes) for a mean difference curve (tauV - tau only).

    kappa* is the smallest grid kappa at which the difference is > 0 there and at
    every larger grid kappa (a stable up-crossing; a tie at 0 is not > 0).
      crossing    kappa* exists and is above the grid minimum
      below_grid  diff > 0 everywhere: tauV leads throughout (kappa* = NaN)
      above_grid  diff <= 0 at the grid maximum: tau only still leads (kappa* = NaN)
    n_sign_changes counts changes of (diff > 0) across consecutive grid points."""
    pos = np.asarray(mean_diff, dtype=float) > 0
    grid = np.asarray(grid, dtype=float)
    n_changes = int(np.sum(pos[1:] != pos[:-1]))
    if pos.all():
        return float('nan'), BELOW_GRID, n_changes
    if not pos[-1]:
        return float('nan'), ABOVE_GRID, n_changes
    last_nonpos = int(np.max(np.nonzero(~pos)[0]))
    return float(grid[last_nonpos + 1]), CROSSING, n_changes


def crossover_kappa(mean_diff, grid):
    """kappa* (see classify_kappa_star); NaN unless the outcome is 'crossing'."""
    return classify_kappa_star(mean_diff, grid)[0]


def bootstrap_crossover(diff, grid, n_boot=N_BOOT, seed=0):
    """Percentile bootstrap of kappa* over replications.

    diff: (reps, K) per-rep welfare difference (tauV minus tau only).  Every
    resample is classified crossing / below_grid / above_grid and all three are
    counted; none is dropped.  ci95 is UNCONDITIONAL: below_grid resamples sort as
    the grid minimum and above_grid as the grid maximum, and ci95_labels marks a
    bound that is such a censored value ('<min' / '>max', else None).
    ci95_conditional_on_crossing uses the crossing resamples only (None if none).
    frac_with_crossover = n_crossing / n_boot."""
    diff = np.asarray(diff, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n = len(diff)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, (n_boot, n))
    cls = [classify_kappa_star(diff[row].mean(axis=0), grid) for row in idx]
    outcomes = np.array([c[1] for c in cls])
    stars = np.array([c[0] for c in cls])
    values = np.where(outcomes == BELOW_GRID, grid.min(),
                      np.where(outcomes == ABOVE_GRID, grid.max(), stars))
    rank = np.array([_OUTCOME_RANK[o] for o in outcomes])
    order = np.lexsort((rank, values))          # by value, ties: below < crossing < above
    sv, so = values[order], outcomes[order]
    i_lo, i_hi = int(np.floor(0.025 * (n_boot - 1))), int(np.ceil(0.975 * (n_boot - 1)))
    label = {BELOW_GRID: '<min', ABOVE_GRID: '>max', CROSSING: None}
    n_cross = int(np.sum(outcomes == CROSSING))
    res = {'n_boot': n_boot, 'n_crossing': n_cross,
           'n_below_grid': int(np.sum(outcomes == BELOW_GRID)),
           'n_above_grid': int(np.sum(outcomes == ABOVE_GRID)),
           'frac_with_crossover': n_cross / n_boot,
           'median': float(np.median(values)),
           'ci95': [float(sv[i_lo]), float(sv[i_hi])],
           'ci95_labels': [label[so[i_lo]], label[so[i_hi]]],
           'ci95_conditional_on_crossing': None}
    res['n_no_crossover'] = n_boot - n_cross
    if n_cross:
        c = stars[outcomes == CROSSING]
        res['ci95_conditional_on_crossing'] = [float(np.percentile(c, 2.5)),
                                               float(np.percentile(c, 97.5))]
    return res


def paired_inference(w_tauv, w_tau, grid):
    """Per-kappa paired t-test of tauV vs tau only (core.stats.paired_test_raw),
    Holm across all kappa.  Everything is unrounded."""
    w_tauv, w_tau = np.asarray(w_tauv, float), np.asarray(w_tau, float)
    rows = []
    for j, kappa in enumerate(grid):
        a, b = w_tauv[:, j], w_tau[:, j]
        r = paired_test_raw(a, b)
        rows.append({'kappa': float(kappa), 'tauV': _t_ci(a), 'tau_only': _t_ci(b),
                     'diff': _t_ci(a - b), 'se_diff': r['se_diff'], 't_stat': r['t_stat'],
                     'cohens_d': r['cohens_d'], 'p_raw': r['p_value']})
    for row, p_holm in zip(rows, holm_bonferroni([r['p_raw'] for r in rows])):
        row['p_holm'] = float(p_holm)
    return rows


def summarise(raw, grid=budget_grid, n_boot=N_BOOT):
    """Per-cell per-kappa welfare + Holm p-values + kappa* (point, CI, frac)."""
    boot_seed = seed_from(SEED_BASE, BOOT_STREAM)
    cells = {}
    for key, w in raw.items():
        diff = w['tauV'] - w['tau']
        mean_diff = diff.mean(axis=0)
        point, outcome, n_changes = classify_kappa_star(mean_diff, grid)
        cells[key] = {
            'per_kappa': paired_inference(w['tauV'], w['tau'], grid),
            'kappa_star': {'point': None if np.isnan(point) else point,
                           'point_outcome': outcome, 'n_sign_changes': n_changes,
                           **bootstrap_crossover(diff, grid, n_boot, boot_seed)},
        }
    return {'cells': cells, 'kappa_grid': [float(k) for k in grid]}


def run_budget_inference(dgp_name, label, reps, n_players=N_PLAYERS):
    """The legacy v44 budget table (T-learner tau_hat, misspecified V) for one DGP.

    Same cell as the paper's CATE only vs CATE x LTV comparison; Holm runs across
    all kappa of the grid, not the four reported budgets."""
    print(f"  [A] budget inference + crossover ({label})")
    raw = simulate(reps, n_players, dgp_name, tau_sources=('T',), v_sources=('misspec',))
    cell = summarise(raw)['cells'][cell_key('T', 'misspec')]
    per_budget = {}
    for row in cell['per_kappa']:
        if round(row['kappa'], 3) in REPORTED_BUDGETS:
            per_budget[str(round(row['kappa'], 3))] = {
                'cate_only_welfare': row['tau_only']['mean'],
                'cate_x_ltv_welfare': row['tauV']['mean'],
                'diff_ltv_minus_cate': row['diff']['mean'],
                'se_diff': row['se_diff'], 't_stat': row['t_stat'],
                'cohens_d': row['cohens_d'],
                'ci_95': row['diff']['ci95'], 'p_value': row['p_raw'],
                'p_holm': row['p_holm'],
            }
    return {'per_budget': per_budget, 'kappa_star': cell['kappa_star'],
            'kappa_grid': [float(k) for k in budget_grid],
            'mean_diff_curve': [r['diff']['mean'] for r in cell['per_kappa']]}


def _print_table(summary):
    print(f"{'cell (tau|V)':<16} {'kappa*':>10} {'95% CI':>17} {'cross':>5} {'<grid':>5} {'>grid':>5} {'chg':>3}  min p_holm")
    for key, c in summary['cells'].items():
        ks = c['kappa_star']
        pt = ks['point_outcome'] if ks['point'] is None else f"{ks['point']:.3f}"
        lab = [l or '' for l in ks['ci95_labels']]
        ci = f"[{lab[0]}{ks['ci95'][0]:.3f},{lab[1]}{ks['ci95'][1]:.3f}]"
        p = min(r['p_holm'] for r in c['per_kappa'])
        print(f"{key:<16} {pt:>10} {ci:>17} {ks['n_crossing']:>5} {ks['n_below_grid']:>5} "
              f"{ks['n_above_grid']:>5} {ks['n_sign_changes']:>3}  {p:.4g}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_common_args(parser)
    args = parser.parse_args(argv)
    reps, n_players = resolve_run_size(args, N_RUNS, N_PLAYERS)
    print(f"Budget 2x2 | reps={reps}, n={n_players}, "
          f"{len(TAU_SOURCES) * len(V_SOURCES)} cells x {len(budget_grid)} kappa")
    summary = summarise(simulate(reps, n_players))
    summary['_meta'] = {'n_runs': reps, 'n_players': n_players, 'seeds_base': SEED_BASE,
                        'n_boot': N_BOOT, 'tau_sources': list(TAU_SOURCES),
                        'v_sources': list(V_SOURCES), 'dgp': 'main'}
    _print_table(summary)
    print(f"Saved {save_json(summary, 'budget_2x2.json')}")


if __name__ == '__main__':
    main()
