"""CATE learner (T/X/S/DR/R) x value model comparison with frozen tuning, plus an
order-free regret split.

Replaces the legacy add-on scripts (the telescoping diagnostic and the end-to-end
model both live here now) and compute_depth_ablation.py, whose depth-matched
S/T comparison is superseded by the tuned comparison below.  (The like-for-like feature
drop now lives in study_robustness.py.)

Learners compared: T, X, S (legacy fits) and DR (cross-fitted, e = 0.5 known; see
core.estimators), each with "default" params (the legacy fit for T/X/S, a depth-3
100-tree fit for DR) and "tuned" params.  The R-learner (estimate_cate_r) is NOT in
the comparison: with the propensity known and equal to 0.5, DR and R are
algebraically the same estimator up to how E[Y|X] versus mu0, mu1 are estimated
(2(W-1/2)(Y-m) and the AIPW pseudo-outcome have the same conditional mean, and R's
weights (W-e)^2 are constant), and their tuned R-losses agree to six decimals;
tests/test_estimators.py checks that their tau_hat correlate above 0.99.  Value models: misspec (the deliberately
misspecified proxy) and wellspec.  Rules: tau_hat * V_hat ("tauV") and tau_hat
alone ("tau"), which ignores V and so is scored once per learner with no value-model
dimension.  Every learner and value model is fitted ONCE per replication and
reused across rules, kappas, the regret split and the diagnostic.  "E2E" is the
end-to-end model of the legacy add-on (one model trained on V_proxy * Y, scored by the
treated-minus-control difference); its definition is unchanged and it is not tuned.
Welfare is always evaluated on the realised TRUE tau and V of the test split.
"oracle" always means the feasible E[tau|X], E[V|X] of core.dgp.conditional_means.

Tuning (frozen in the tracked config/tuned_params.json): grid max_depth {1,2,3,4} x
n_estimators {25,50,100,300} x learning_rate {0.005,0.01,0.02,0.05,0.1} (80 configs).
The first 48-config grid (trees from 50, rate from 0.01) put T, X and DR at the trees=50
and/or rate=0.01 edge, so both axes were extended one step (trees 25, rate 0.005),
once.  Each learner's optimum is checked against the grid edges and the check is
recorded in the file's _meta.  Per learner, a pilot of 5 replications with
seeds 10_000 + 7r (disjoint from the evaluation seeds 42 + 7r, r < 200) is run; in
each, the training split is cross-fitted into 2 folds, the learner is fitted on one
fold and scored on the other by the R-loss
mean(((Y - m_hat) - (W - e) * tau_hat)^2), where m_hat = E[Y|X] is fitted on the
fitting fold (default GB classifier, shared by all configs).  The config with the
lowest mean loss wins (ties go to the earlier grid entry).  The pilot runs only if
the frozen file is missing or --retune is given; the main run reads the file and
raises if its _meta (grid, pilot seeds, n, dgp) differs from the current code.  The
tracked file is full-size only: other run sizes (including --smoke) use
outputs/tuned_params_n<N>.json and never touch config/.

Inference: for every kappa, paired_test_raw of each (learner, params, value model,
rule) welfare against "T (default)" under the same value model and rule (tau-only
against "T (default)" tau-only), and of E2E against "T (default)" with the misspec
value model and rule tauV.  Holm
(core.stats.holm_bonferroni) runs across ALL those comparisons within a kappa (the
family is every comparison at that kappa).  Everything is unrounded in the JSON.

Regret split (kappa = 0.30): core.policy.regret_split against the oracle
(tau_bar, ltv_bar), per learner x params x value model: Shapley CATE/value shares
that do not depend on the telescoping order, plus both single-path splits.

Diagnostic: over the misranked set M = oracle top-k XOR estimated top-k,
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

from study_main import N_PLAYERS, N_RUNS, SEED_BASE
from core.dgp import DGPS, FEATURE_NAMES, conditional_means
from core.estimators import (
    CROSSFIT_STREAM, DEFAULT_PARAMS, LTV_STREAM, ORACLE_STREAM,
    _estimate_cate_tlearner, _estimate_ltv_proxy,
    _wellspec_value_model, estimate_cate, seed_from,
)
from pathlib import Path

from core.io import OUTPUT_DIR, add_common_args, resolve_run_size, save_json
from core.policy import DEFAULT_BUDGET, PROPENSITY, compute_welfare, regret_split, target_topk
from core.stats import holm_bonferroni, paired_test_raw, t_mean_ci

KAPPAS = (0.10, 0.20, 0.30, 0.50)
REGRET_KAPPA = DEFAULT_BUDGET
PSETS = ('tuned', 'default')
VALUE_MODELS = ('misspec', 'wellspec')
COMPARED_LEARNERS = ('T', 'X', 'S', 'DR')   # R is algebraically DR at e = 0.5 (docstring)
ORACLE_MC = 200
BASELINE = 'T (default)'
BASELINE_CELL = f'{BASELINE}|misspec'
E2E_KEY = 'E2E'
LOGIN_COL = FEATURE_NAMES.index('login_freq')
CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'

GRID = [{'max_depth': d, 'n_estimators': n, 'learning_rate': lr}
        for d, n, lr in itertools.product((1, 2, 3, 4), (25, 50, 100, 300),
                                          (0.005, 0.01, 0.02, 0.05, 0.1))]
PILOT_REPS = 5
PILOT_SEED_BASE = 10_000
TUNED_FILE = 'tuned_params.json'


def pilot_seeds(n_reps=PILOT_REPS):
    """Pilot seeds 10_000 + 7r; never equal to an evaluation seed SEED_BASE + 7r' (r' < 200)."""
    return [PILOT_SEED_BASE + 7 * r for r in range(n_reps)]


def label(learner, pset):
    return f'{learner} ({pset})'


def combo_key(learner, pset, vm=None):
    """'L (pset)|vm' is the tau_hat * V_hat rule; 'L (pset)|tau' is tau_hat alone
    (it ignores V, so it has no value-model dimension)."""
    return f'{label(learner, pset)}|{vm if vm else "tau"}'


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


def boundary_check(grid, params):
    """Per axis: 'floor' / 'ceiling' if params sits at the grid minimum / maximum of
    that axis, else 'interior'.  A depth-1 floor is reported as 'hard_floor' (no
    shallower tree exists); trees and rate floors are extendable axes."""
    out = {}
    for axis in ('max_depth', 'n_estimators', 'learning_rate'):
        vals = sorted({g[axis] for g in grid})
        if params[axis] == vals[0]:
            out[axis] = 'hard_floor' if axis == 'max_depth' and vals[0] == 1 else 'floor'
        elif params[axis] == vals[-1]:
            out[axis] = 'ceiling'
        else:
            out[axis] = 'interior'
    return out


def make_meta(n_players, grid=GRID, seeds=None, dgp_name='main'):
    seeds = pilot_seeds() if seeds is None else seeds
    return {'grid': grid, 'pilot_seeds': seeds, 'n_players': n_players, 'dgp': dgp_name,
            'learners': list(COMPARED_LEARNERS),
            'criterion': 'mean cross-fitted R-loss, 2 folds of the pilot training split'}


def tune_all(n_players=N_PLAYERS, n_jobs=1, grid=GRID, seeds=None, dgp_name='main'):
    out = {'learners': {}, '_meta': make_meta(n_players, grid, seeds, dgp_name)}
    seeds = out['_meta']['pilot_seeds']
    for learner in COMPARED_LEARNERS:
        t0 = time.perf_counter()
        res = tune_learner(learner, grid, seeds, n_players, dgp_name, n_jobs)
        out['learners'][learner] = res
        out['_meta'].setdefault('boundary_check', {})[learner] = boundary_check(grid, res['params'])
        print(f"  tuned {learner:>2}: {res['params']} R-loss={res['mean_loss']:.6f} "
              f"edges={out['_meta']['boundary_check'][learner]} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return out


def tuned_path(n_players):
    """Tracked config file for the full-size pilot; an untracked outputs/ file for any
    other run size (smoke included), so a small run never overwrites the config."""
    if n_players == N_PLAYERS:
        return CONFIG_DIR / TUNED_FILE
    return OUTPUT_DIR / f'tuned_params_n{n_players}.json'


def _check_meta(frozen_meta, expected, path):
    for key in ('grid', 'pilot_seeds', 'n_players', 'dgp'):
        if frozen_meta.get(key) != expected[key]:
            raise ValueError(
                f"{path}: frozen tuning _meta[{key!r}] = {frozen_meta.get(key)!r} differs from "
                f"the current code's {expected[key]!r}; rerun with --retune")


def load_or_tune(n_players=N_PLAYERS, retune=False, n_jobs=1):
    """Frozen tuned params {learner: params} for runs of size n_players.

    Runs the pilot (at n_players) only if the file is missing or retune is set;
    otherwise reads it and raises ValueError if its _meta (grid, pilot seeds, n,
    dgp) differs from the current code."""
    path = tuned_path(n_players)
    if retune or not path.exists():
        print(f"Tuning pilot at n={n_players}, writing {path}")
        frozen = tune_all(n_players, n_jobs)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as fp:
            json.dump(frozen, fp, indent=2)
    else:
        print(f"Reading frozen tuned params from {path}")
        frozen = json.loads(path.read_text())
        _check_meta(frozen['_meta'], make_meta(n_players), path)
    return {l: frozen['learners'][l]['params'] for l in COMPARED_LEARNERS}


# ---------------------------------------------------------------------------
# End-to-end model and diagnostic
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
def run_rep(rep, n_players, tuned, dgp_name='main', learners=COMPARED_LEARNERS, kappas=KAPPAS):
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
            welfare[combo_key(learner, pset, vm)] = [
                welfare_at_kappa(tau_hat * vs[vm], kp, true_tau, true_ltv) for kp in kappas]
        welfare[combo_key(learner, pset)] = [
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


def simulate(reps, n_players, tuned, dgp_name='main', learners=COMPARED_LEARNERS, kappas=KAPPAS,
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
    """(key, baseline key) for every non-baseline combo, same value model (tau-only against
    tau-only); E2E against BASELINE_CELL."""
    pairs = []
    for key in keys:
        if key == E2E_KEY:
            pairs.append((key, BASELINE_CELL))
            continue
        base = f'{BASELINE}|{key.split("|")[1]}'
        if key != base:                      # a baseline is not compared with itself
            pairs.append((key, base))
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
    tuned = load_or_tune(n_players=n_players, retune=args.retune, n_jobs=args.jobs)
    summary = summarise(simulate(reps, n_players, tuned, n_jobs=args.jobs))
    summary['_meta'] = {'n_runs': reps, 'n_players': n_players, 'seeds_base': SEED_BASE,
                        'kappas': list(KAPPAS), 'tuned_params': tuned, 'dgp': 'main'}
    _print_table(summary)
    print(f"Saved {save_json(summary, 'learners_smoke.json' if args.smoke else 'learners.json')}")


if __name__ == '__main__':
    main()
