"""CATE learner (T/X/S/DR/R) x value model comparison with frozen tuning, plus an
order-free regret split.

Replaces compute_v41_additions.py (the telescoping diagnostic and the end-to-end
model both live here now) and compute_depth_ablation.py, whose depth-matched
S/T comparison is superseded by the tuned comparison below.  (compute_v45_additions.py,
the like-for-like feature drop, is untouched here.)

Learners: T, X, S (legacy fits) and DR, R (cross-fitted, e = 0.5 known; see
core.estimators), each with "default" params (the legacy fit for T/X/S, a depth-3
100-tree fit for DR/R) and "tuned" params.  Value models: misspec (the deliberately
misspecified proxy) and wellspec.  Rules: tau_hat * V_hat ("tauV") and tau_hat
alone ("tau").  Every learner and value model is fitted ONCE per replication and
reused across rules, kappas, the regret split and the diagnostic.  "E2E" is the
end-to-end model of v41 (one model trained on V_proxy * Y, scored by the
treated-minus-control difference); its definition is unchanged and it is not tuned.
Welfare is always evaluated on the realised TRUE tau and V of the test split.
"oracle" always means the feasible E[tau|X], E[V|X] of core.dgp.conditional_means.

Tuning (frozen in outputs/tuned_params.json): grid max_depth {2,3,4} x n_estimators
{100,300} x learning_rate {0.05,0.1}.  Per learner, a pilot of 5 replications with
seeds 10_000 + 7r (disjoint from the evaluation seeds 42 + 7r, r < 200) is run; in
each, the training split is cross-fitted into 2 folds, the learner is fitted on one
fold and scored on the other by the R-loss
mean(((Y - m_hat) - (W - e) * tau_hat)^2), where m_hat = E[Y|X] is fitted on the
fitting fold (default GB classifier, shared by all configs).  The config with the
lowest mean loss wins (ties go to the earlier grid entry).  The pilot runs only if
the frozen file is missing or --retune is given; the main run reads the file.

Inference: for every kappa, paired_test_raw of each (learner, params, value model,
rule) welfare against "T (default)" under the same value model and rule, and of E2E
against "T (default)" with the misspec value model and rule tauV.  Holm
(core.stats.holm_bonferroni) runs across ALL those comparisons within a kappa (the
family is every comparison at that kappa).  Everything is unrounded in the JSON.

Regret split (kappa = 0.30): core.policy.regret_split against the oracle
(tau_bar, ltv_bar), per learner x params x value model: Shapley CATE/value shares
that do not depend on the telescoping order, plus both single-path splits.

Diagnostic (from v41): over the misranked set M = oracle top-k XOR estimated top-k,
the telescoping bound terms |tau - tau_hat| |V| (CATE) and |tau_hat| |V - V_hat|
(value) and the CATE share, for the full-feature T-learner and for one with login
dropped from its features (CATE misspecification); V_hat is the misspec proxy.

Usage:  python study_learners.py [--reps R] [--n-players N] [--smoke] [--retune] [--jobs J]
"""

import argparse
import itertools
import json
import time
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split

from simulation_v38 import N_PLAYERS, N_RUNS, SEED_BASE
from core.dgp import DGPS, FEATURE_NAMES, conditional_means
from core.estimators import (
    CROSSFIT_STREAM, DEFAULT_PARAMS, LEARNERS, LTV_STREAM, ORACLE_STREAM,
    _estimate_cate_tlearner, _estimate_ltv_proxy,
    _wellspec_value_model, estimate_cate, seed_from,
)
from core.io import OUTPUT_DIR, SMOKE_N_PLAYERS, add_common_args, resolve_run_size, save_json
from core.policy import DEFAULT_BUDGET, PROPENSITY, compute_welfare, regret_split, target_topk
from core.stats import holm_bonferroni, paired_test_raw, t_mean_ci

KAPPAS = (0.10, 0.20, 0.30, 0.50)
REGRET_KAPPA = DEFAULT_BUDGET
PSETS = ('tuned', 'default')
VALUE_MODELS = ('misspec', 'wellspec')
RULES = ('tauV', 'tau')
ORACLE_MC = 200
BASELINE = 'T (default)'
BASELINE_CELL = f'{BASELINE}|misspec|tauV'
E2E_KEY = 'E2E'
LOGIN_COL = FEATURE_NAMES.index('login_freq')

GRID = [{'max_depth': d, 'n_estimators': n, 'learning_rate': lr}
        for d, n, lr in itertools.product((2, 3, 4), (100, 300), (0.05, 0.1))]
PILOT_REPS = 5
PILOT_SEED_BASE = 10_000
TUNED_FILE = 'tuned_params.json'
TUNED_FILE_SMOKE = 'tuned_params_smoke.json'


def pilot_seeds(n_reps=PILOT_REPS):
    """Pilot seeds 10_000 + 7r; never equal to an evaluation seed SEED_BASE + 7r' (r' < 200)."""
    return [PILOT_SEED_BASE + 7 * r for r in range(n_reps)]


def label(learner, pset):
    return f'{learner} ({pset})'


def combo_key(learner, pset, vm, rule):
    return f'{label(learner, pset)}|{vm}|{rule}'


def welfare_at_kappa(scores, kappa, true_tau, true_ltv):
    """True welfare of the top-k set by `scores`, k = int(n * kappa)."""
    return compute_welfare(target_topk(scores, int(len(scores) * kappa)), true_tau, true_ltv)


def _split(data, seed):
    idx = np.arange(len(data['treatment']))
    return train_test_split(idx, test_size=0.30, random_state=seed)


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
def _pilot_rep_losses(learner, grid, seed, n_players, dgp_name):
    """Per-config cross-fitted R-loss on one pilot replication (mean over 2 folds)."""
    data = DGPS[dgp_name](n_players, np.random.RandomState(seed))
    train_idx, _ = _split(data, seed)
    X, W, Y = (data['features'][train_idx], data['treatment'][train_idx],
               data['y_obs'][train_idx])
    folds = list(KFold(n_splits=2, shuffle=True,
                       random_state=seed_from(seed, CROSSFIT_STREAM)).split(np.arange(len(Y))))
    # m_hat on the held-out fold comes from a model fitted on the fitting fold only.
    m_held = []
    for fit, held in folds:
        m = GradientBoostingClassifier(random_state=0, **DEFAULT_PARAMS['R'])
        m.fit(X[fit], Y[fit])
        m_held.append(m.predict_proba(X[held])[:, 1])
    losses = []
    for params in grid:
        fold_loss = []
        for (fit, held), m_hat in zip(folds, m_held):
            tau_hat = estimate_cate(learner, X[fit], X[held], W[fit], Y[fit], params, seed=seed)
            resid = (Y[held] - m_hat) - (W[held] - PROPENSITY) * tau_hat
            fold_loss.append(float(np.mean(resid ** 2)))
        losses.append(float(np.mean(fold_loss)))
    return losses


def tune_learner(learner, grid=GRID, seeds=None, n_players=N_PLAYERS, dgp_name='main',
                 n_jobs=1):
    """Select params for one learner by mean cross-fitted R-loss over the pilot reps.

    Deterministic: ties go to the earliest grid entry.  Returns
    {'params', 'mean_loss', 'losses': [mean loss per grid entry]}."""
    seeds = pilot_seeds() if seeds is None else seeds
    assert not set(seeds) & {SEED_BASE + 7 * r for r in range(N_RUNS)}, \
        'pilot seeds overlap evaluation seeds'
    per_rep = Parallel(n_jobs=n_jobs)(
        delayed(_pilot_rep_losses)(learner, grid, s, n_players, dgp_name) for s in seeds)
    losses = np.mean(per_rep, axis=0)
    best = int(np.argmin(losses))
    return {'params': dict(grid[best]), 'mean_loss': float(losses[best]),
            'losses': [float(x) for x in losses]}


def tune_all(n_players=N_PLAYERS, n_jobs=1, grid=GRID, seeds=None, dgp_name='main'):
    seeds = pilot_seeds() if seeds is None else seeds
    out = {'learners': {}, '_meta': {
        'grid': grid, 'pilot_seeds': seeds, 'n_players': n_players, 'dgp': dgp_name,
        'criterion': 'mean cross-fitted R-loss, 2 folds of the pilot training split'}}
    for learner in LEARNERS:
        t0 = time.perf_counter()
        out['learners'][learner] = tune_learner(learner, grid, seeds, n_players, dgp_name, n_jobs)
        print(f"  tuned {learner:>2}: {out['learners'][learner]['params']} "
              f"R-loss={out['learners'][learner]['mean_loss']:.6f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return out


def load_or_tune(smoke=False, retune=False, n_players=N_PLAYERS, n_jobs=1):
    """Frozen tuned params {learner: params}; runs the pilot only if the file is
    missing or retune is set.  Smoke runs use their own file and pilot size."""
    name = TUNED_FILE_SMOKE if smoke else TUNED_FILE
    path = OUTPUT_DIR / name
    if retune or not path.exists():
        print(f"Tuning pilot ({'smoke size' if smoke else 'full size'}), writing {name}")
        frozen = tune_all(SMOKE_N_PLAYERS if smoke else n_players, n_jobs)
        save_json(frozen, name)
    else:
        print(f"Reading frozen tuned params from {path}")
        frozen = json.loads(path.read_text())
    return {l: frozen['learners'][l]['params'] for l in LEARNERS}


# ---------------------------------------------------------------------------
# End-to-end model and diagnostic (from v41)
# ---------------------------------------------------------------------------
def end_to_end_score(Xtr, Xte, Wtr, Ytr, seed):
    """E2E: one model trained on V_proxy * Y with treatment as a feature, scored by the
    treated-minus-control difference.  Uses the same misspecified value-proxy form as
    the decomposed pipeline's training target (not tuned)."""
    rng_v = np.random.RandomState(seed_from(seed, LTV_STREAM))
    vproxy = (12 * Xtr[:, 1] ** 0.9 + 35 * Xtr[:, 2]
              + rng_v.normal(0, 200, len(Xtr))).clip(50, 8000)
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=0)
    model.fit(np.column_stack([Xtr, Wtr]), vproxy * Ytr)
    n = len(Xte)
    return (model.predict(np.column_stack([Xte, np.ones(n)]))
            - model.predict(np.column_stack([Xte, np.zeros(n)])))


def diagnostic_rep(tau_hat, v_hat, tau, v, k):
    """(CATE term, value term, |M|) over the misranked set, or None if M is empty."""
    pi_star = set(target_topk(tau * v, k).tolist())
    pi_hat = set(target_topk(tau_hat * v_hat, k).tolist())
    m = np.array(sorted(pi_star.symmetric_difference(pi_hat)), dtype=int)
    if len(m) == 0:
        return None
    return (float((np.abs(tau[m] - tau_hat[m]) * np.abs(v[m])).mean()),
            float((np.abs(tau_hat[m]) * np.abs(v[m] - v_hat[m])).mean()), len(m))


# ---------------------------------------------------------------------------
# One replication, simulation
# ---------------------------------------------------------------------------
def run_rep(rep, n_players, tuned, dgp_name='main', learners=LEARNERS, kappas=KAPPAS):
    """Fit everything once for one replication and score all combos."""
    seed = SEED_BASE + rep * 7
    t = defaultdict(float)
    tick = time.perf_counter()

    def lap(name):
        nonlocal tick
        now = time.perf_counter()
        t[name] += now - tick
        tick = now

    data = DGPS[dgp_name](n_players, np.random.RandomState(seed))
    train_idx, test_idx = _split(data, seed)
    true_tau, true_ltv = data['true_tau'][test_idx], data['true_ltv'][test_idx]
    X, W, Y = data['features'], data['treatment'], data['y_obs']
    Xtr, Xte, Wtr, Ytr = X[train_idx], X[test_idx], W[train_idx], Y[train_idx]
    lap('data')

    taus = {}
    for learner in learners:
        for pset in PSETS:
            params = tuned[learner] if pset == 'tuned' else None
            taus[(learner, pset)] = estimate_cate(learner, Xtr, Xte, Wtr, Ytr, params, seed=seed)
            lap(f'fit_{learner}')
    vs = {'misspec': _estimate_ltv_proxy(Xtr, Xte, seed),
          'wellspec': _wellspec_value_model(Xtr, Xte, seed)}
    lap('value_models')
    feats = {name: X[test_idx, i] for i, name in enumerate(FEATURE_NAMES)}
    cm = conditional_means(dgp_name, feats, n_mc=ORACLE_MC, seed=seed_from(seed, ORACLE_STREAM))
    tau_bar, ltv_bar = cm['tau_bar'], cm['ltv_bar']
    lap('oracle')
    e2e = end_to_end_score(Xtr, Xte, Wtr, Ytr, seed)
    lap('e2e')

    welfare = {}
    for (learner, pset), tau_hat in taus.items():
        for vm in VALUE_MODELS:
            welfare[combo_key(learner, pset, vm, 'tauV')] = [
                welfare_at_kappa(tau_hat * vs[vm], kp, true_tau, true_ltv) for kp in kappas]
            welfare[combo_key(learner, pset, vm, 'tau')] = [
                welfare_at_kappa(tau_hat, kp, true_tau, true_ltv) for kp in kappas]
    welfare[E2E_KEY] = [welfare_at_kappa(e2e, kp, true_tau, true_ltv) for kp in kappas]
    lap('welfare')

    k = int(len(test_idx) * REGRET_KAPPA)
    regret = {f'{label(l, p)}|{vm}': regret_split(tau_bar, ltv_bar, tau_hat, vs[vm],
                                                  true_tau, true_ltv, k)
              for (l, p), tau_hat in taus.items() for vm in VALUE_MODELS}
    lap('regret')

    diag = {}
    if ('T', 'default') in taus:
        cols = [i for i in range(Xtr.shape[1]) if i != LOGIN_COL]
        tau_drop = _estimate_cate_tlearner(Xtr[:, cols], Xte[:, cols], Wtr, Ytr)
        diag = {'cate_full': diagnostic_rep(taus[('T', 'default')], vs['misspec'],
                                            true_tau, true_ltv, k),
                'cate_login_dropped': diagnostic_rep(tau_drop, vs['misspec'],
                                                     true_tau, true_ltv, k)}
    lap('diagnostic')
    return {'welfare': welfare, 'regret': regret, 'diagnostic': diag, 'timing': dict(t)}


def simulate(reps, n_players, tuned, dgp_name='main', learners=LEARNERS, kappas=KAPPAS,
             n_jobs=1):
    """Per-rep welfare for every combo (and E2E), regret splits, diagnostics, timing.

    Returns {'welfare': {key: (reps, K) array}, 'regret': [per-rep dicts],
    'diagnostic': [per-rep dicts], 'timing': [per-rep dicts], 'kappas': [...]}."""
    rows = Parallel(n_jobs=n_jobs)(
        delayed(run_rep)(r, n_players, tuned, dgp_name, learners, kappas) for r in range(reps))
    keys = rows[0]['welfare'].keys()
    return {'welfare': {key: np.array([row['welfare'][key] for row in rows]) for key in keys},
            'regret': [row['regret'] for row in rows],
            'diagnostic': [row['diagnostic'] for row in rows],
            'timing': [row['timing'] for row in rows],
            'kappas': [float(kp) for kp in kappas]}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def comparison_pairs(keys):
    """(key, baseline key) for every non-baseline combo; E2E against BASELINE_CELL."""
    pairs = []
    for key in keys:
        if key == BASELINE_CELL:
            continue
        if key == E2E_KEY:
            pairs.append((key, BASELINE_CELL))
            continue
        _, vm, rule = key.split('|')
        pairs.append((key, f'{BASELINE}|{vm}|{rule}'))
    return pairs


def summarise_welfare(raw):
    """Per-combo mean/CI per kappa, and per kappa the paired test of every combo vs
    "T (default)" with Holm across all comparisons within the kappa."""
    kappas = raw['kappas']
    welfare = raw['welfare']
    pairs = [p for p in comparison_pairs(list(welfare)) if p[1] in welfare]
    means = {key: [t_mean_ci(w[:, j]) for j in range(len(kappas))] for key, w in welfare.items()}
    tests = {}
    for j, kp in enumerate(kappas):
        rows = []
        for key, base in pairs:
            r = paired_test_raw(welfare[key][:, j], welfare[base][:, j])
            rows.append({'combo': key, 'baseline': base, 'diff': r['mean_diff'],
                         'ci95': r['ci_95'], 'se_diff': r['se_diff'], 't_stat': r['t_stat'],
                         'cohens_d': r['cohens_d'], 'p_raw': r['p_value']})
        for row, p in zip(rows, holm_bonferroni([r['p_raw'] for r in rows])):
            row['p_holm'] = float(p)
        tests[str(kp)] = rows
    return {'welfare': means, 'tests_vs_T_default': tests, 'kappas': kappas,
            'n_comparisons_per_kappa': len(pairs)}


def summarise_regret(per_rep):
    """Per learner x params x value model at kappa 0.30: mean and CI over reps of
    phi_tau, phi_v and total; Shapley and single-path shares of the mean total
    (stable) and the mean of per-rep Shapley shares over non-degenerate reps."""
    out = {}
    for key in per_rep[0]:
        rows = [r[key] for r in per_rep]
        total = float(np.mean([r['total'] for r in rows]))
        entry = {'phi_tau': t_mean_ci([r['phi_tau'] for r in rows]),
                 'phi_v': t_mean_ci([r['phi_v'] for r in rows]),
                 'total': t_mean_ci([r['total'] for r in rows]),
                 'n_zero_denominator': int(sum(r['zero_denominator'] for r in rows))}
        for order in ('order_a', 'order_b'):
            tau_part = float(np.mean([r[order]['tau'] for r in rows]))
            entry[f'{order}_tau_share_of_mean'] = tau_part / total if total else float('nan')
        entry['shapley_tau_share_of_mean'] = (entry['phi_tau']['mean'] / total
                                              if total else float('nan'))
        entry['shapley_v_share_of_mean'] = (entry['phi_v']['mean'] / total
                                            if total else float('nan'))
        shares = [r['shapley']['tau_share'] for r in rows if not r['zero_denominator']]
        entry['shapley_tau_share_per_rep'] = t_mean_ci(shares) if shares else None
        out[key] = entry
    return out


def summarise_diagnostic(per_rep):
    out = {}
    for name in ('cate_full', 'cate_login_dropped'):
        terms = [r[name] for r in per_rep if r.get(name)]
        if not terms:
            continue
        a, b = (float(np.mean([t[i] for t in terms])) for i in (0, 1))
        out[name] = {'cate_error_term': a, 'value_error_term': b,
                     'cate_share_pct': 100 * a / (a + b),
                     'mean_misranked_set_size': float(np.mean([t[2] for t in terms])),
                     'n_reps': len(terms)}
    return out


def summarise_timing(per_rep):
    stages = {s for t in per_rep for s in t}
    mean = {s: float(np.mean([t.get(s, 0.0) for t in per_rep])) for s in sorted(stages)}
    return {'mean_seconds_per_rep': float(sum(mean.values())), 'by_stage': mean}


def summarise(raw):
    return {**summarise_welfare(raw), 'regret_split_kappa_0.30': summarise_regret(raw['regret']),
            'diagnostic': summarise_diagnostic(raw['diagnostic']),
            'timing': summarise_timing(raw['timing'])}


def _print_table(summary):
    j = summary['kappas'].index(REGRET_KAPPA)
    tests = {r['combo']: r for r in summary['tests_vs_T_default'][str(REGRET_KAPPA)]}
    print(f"\nWelfare at kappa={REGRET_KAPPA} (true tau, V); diff vs {BASELINE_CELL}, Holm over "
          f"{summary['n_comparisons_per_kappa']} comparisons")
    print(f"{'combo':<34} {'welfare':>10} {'diff':>9} {'p_holm':>8}")
    for key, cis in summary['welfare'].items():
        t = tests.get(key)
        d = f"{t['diff']:>9.0f} {t['p_holm']:>8.3f}" if t else f"{'-':>9} {'-':>8}"
        print(f"{key:<34} {cis[j]['mean']:>10.0f} {d}")
    print(f"\nRegret split at kappa={REGRET_KAPPA} (Shapley share of mean total regret)")
    print(f"{'learner x value model':<30} {'total':>8} {'CATE share':>11} {'V share':>8} "
          f"{'order_a':>8} {'order_b':>8}")
    for key, r in summary['regret_split_kappa_0.30'].items():
        print(f"{key:<30} {r['total']['mean']:>8.0f} {r['shapley_tau_share_of_mean']:>11.2f} "
              f"{r['shapley_v_share_of_mean']:>8.2f} {r['order_a_tau_share_of_mean']:>8.2f} "
              f"{r['order_b_tau_share_of_mean']:>8.2f}")
    for name, d in summary['diagnostic'].items():
        print(f"diagnostic [{name}] CATE-term={d['cate_error_term']:.0f} "
              f"value-term={d['value_error_term']:.0f} CATE share={d['cate_share_pct']:.1f}% "
              f"|M|~{d['mean_misranked_set_size']:.0f}")
    tm = summary['timing']
    print(f"per-rep runtime {tm['mean_seconds_per_rep']:.1f}s: "
          + ', '.join(f'{s} {v:.1f}' for s, v in tm['by_stage'].items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_common_args(parser)
    parser.add_argument('--retune', action='store_true',
                        help='rerun the tuning pilot even if the frozen file exists')
    parser.add_argument('--jobs', type=int, default=1, help='parallel worker processes')
    args = parser.parse_args(argv)
    reps, n_players = resolve_run_size(args, N_RUNS, N_PLAYERS)
    print(f"CATE learner x value model | reps={reps}, n={n_players}")
    tuned = load_or_tune(smoke=args.smoke, retune=args.retune, n_jobs=args.jobs)
    summary = summarise(simulate(reps, n_players, tuned, n_jobs=args.jobs))
    summary['_meta'] = {'n_runs': reps, 'n_players': n_players, 'seeds_base': SEED_BASE,
                        'kappas': list(KAPPAS), 'tuned_params': tuned, 'dgp': 'main'}
    _print_table(summary)
    print(f"Saved {save_json(summary, 'learners_smoke.json' if args.smoke else 'learners.json')}")


if __name__ == '__main__':
    main()
