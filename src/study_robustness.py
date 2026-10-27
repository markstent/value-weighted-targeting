"""Robustness scenarios for the value-weighted targeting claims.

One function per scenario, each returning a JSON-ready dict; main() runs them all and
writes robustness.json (robustness_smoke.json with --smoke), one top-level key per
scenario.  Absorbs and replaces the legacy compute_* add-on scripts
(the budget and harm-gate sections already live in study_budget.py
and study_harm.py).

Common conventions (all scenarios)
  * Seeds SEED_BASE + 7r, N_PLAYERS 10000, N_RUNS 200, a 70/30 train/test split.  The
    components of one replication are fitted ONCE and reused across every strategy,
    kappa and comparison of that replication.
  * Welfare is always evaluated on the realised TRUE tau and V of the test split.
  * Learner policy: every scenario that fits a CATE learner reports two arms, the
    legacy learner with default params ("CATE Only", "CATE x LTV", ...) and the tuned
    T-learner of config/tuned_params.json (labelled "(tuned)"), loaded through
    study_learners.load_or_tune.  The tuned params were fitted on the MAIN DGP only (the pilot
    of study_learners) and are reused unchanged on every other DGP (null, correlated, asym,
    heavytail, corr_gamma, adv): a fixed-hyperparameter choice, recorded as tuned_on "main"
    in robustness.json _meta.  Only tau_hat changes between arms; the value and
    churn models, the S-learner and the X-learner reference keep their legacy fits.
  * Oracle rows come in two kinds: "Oracle hindsight" ranks on the realised tau * V
    (unreachable by any X-measurable rule) and "Oracle feasible" ranks on
    E[tau|X] E[V|X] from core.dgp.conditional_means (corr_gamma gets gamma and the
    full-sample tau_base moments, as study_main does).  Both use g = 1.  This
    changes the legacy "Oracle hindsight (theta0)" row, which applied the default
    risk modulation to the oracle churn; Proposition 3 makes g = 1 the right oracle.
  * Calibration (asym, effective_welfare, aipw_calibration) is core.policy.
    calibrate_theta_all with the aipw objective at the scenario's evaluation kappa;
    the tuned arm calibrates with the tuned T-learner as its validation-fold CATE.
  * Inference: every reported comparison carries core.stats.paired_test_raw output
    (paired t-test over replications, unrounded) plus p_holm, the Holm-Bonferroni
    adjustment (core.stats.holm_bonferroni) over the scenario's whole family of
    comparisons (every comparison in every cell of that scenario; the size is
    holm_family_size, the definition holm_family).  Nothing stored is rounded.

Scenarios
  slearner_scenarios  CATE x LTV vs CATE Only vs the S-learner on the
      main DGP, the null DGP (no CATE-churn interaction), the correlated-feature DGP
      and the misspec cell (main DGP, login_freq dropped from the T-learner CATE model
      in both arms, S-learner keeping all features; the S-learner is a joint model over
      all features so its welfare there equals the main-cell S-learner by construction,
      but its comparison with the login-dropped decomposition is what the paper
      contrasts).  Comparisons: CATE x LTV vs CATE Only, S-Learner vs CATE x LTV,
      tuned vs default CATE x LTV, Oracle feasible vs CATE x LTV.
  asym  The same strategies on the asymmetric-moderation DGP (tau attenuates
      faster at high churn) plus Full Framework (Est.) (default theta) and Full
      Framework (Cal.) (AIPW-calibrated), with how often calibration chose the
      near-identity config.  Adds Cal. vs CATE x LTV comparisons.
  effective_welfare  Main DGP: welfare scored on nominal tau (the headline) and
      on the realised clip-adjusted tau_eff = y1_prob - y0_prob ("[effective]"), with
      the overstatement ("[overstatement]" = nominal minus effective, per replication).
      Comparisons: nominal vs effective per strategy; each strategy's overstatement
      vs CATE Only's (non-differential bias).  pct_overstate = 100 (nominal -
      effective) / effective of the means, as in the legacy table.
  component_update  Holding tau_hat fixed, swap the misspecified value model for
      a well-specified one; the welfare recovery is the value of in-place component
      updates.  Comparison per arm: well-specified vs misspecified V.
  corr_sweep  V <- V + gamma * 180 * z(tau_base), gamma in {-1, 0, 1}: does the
      value-weighting advantage survive competing value signal?  Reports the sample
      corr(tau, V) and the LTV-weighting advantage in percent.
  heavy_tail  Multiplicative log-normal LTV (orders-of-magnitude spread).
  obs_diagnostic  Plug-in regret diagnostic.  Over the misranked set M (top-k by
      the deployed rule XOR top-k by a reference rule) the telescoping terms are
      |tau_ref - tau_hat| |V_hat| (CATE) and |tau_hat| |V_ref - V_hat| (value).  The
      observable reference is a cross-fitted X-learner tau with a richer value model;
      the same terms against Oracle hindsight (true tau, V) and Oracle feasible
      (tau_bar, V_bar) check that the proxy flags the same component.  Cells: wellspec
      (full-feature T-learner) and misspec (login dropped from the CATE model).
      Comparison per arm and reference: per-replication CATE share (percent) misspec
      vs wellspec, over replications where both have a non-empty M.  The reported
      share (cate_share_pct) is of the replication-mean terms.
  aipw_calibration  Main DGP, kappa 0.30: identity-selection rate and chosen
      configs of the AIPW and the plug-in calibration objectives (the plug-in always
      picks identity: top-k tautology), and the true-welfare gain of each calibrated rule
      over the identity rule (CATE x LTV).  Comparisons: calibrated vs identity.  The two
      plug-in comparisons are exactly zero by construction; they stay in the output
      with "degenerate": true and p_holm None and are NOT in the Holm family.
  adversarial  Adversarial CATE misspecification on the adv DGP (tau also depends
      on bonus_history, which is absent from V): drop bonus_history from the CATE model
      so the value model cannot recover it, at kappa 0.30 and 0.10, with the
      well-specified (all-feature) reference and the all-feature S-learner.
  like_for_like  The S-learner (all features) vs the S-learner missing the SAME
      feature dropped from the decomposed CATE model (login on the main DGP,
      bonus_history on the adv DGP), and the same drop for both T-learner arms.  The
      value model keeps all features throughout.  Comparison: all features vs dropped.

Definition changes against the legacy scripts: the two oracle rows above; the tuned arms;
p-values and Holm everywhere; unrounded storage; the legacy per-seed plug-in/AIPW
cost CI is now the paired t-interval.

Usage:  python study_robustness.py [--reps R] [--n-players N] [--smoke] [--jobs J] [--retune]
"""

import argparse
import time
from collections import Counter
from functools import partial
from types import SimpleNamespace

import numpy as np
from joblib import Parallel, delayed
from scipy import stats as scipy_stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from study_main import N_PLAYERS, N_RUNS, SEED_BASE
from core.dgp import (
    DGPS, FEATURE_NAMES, conditional_means, dgp_spec, generate_adv, make_corr_dgp,
    tau_base_stats,
)
from core.estimators import (
    LTV_STREAM, ORACLE_STREAM, VALUE_STREAM_V43, _estimate_cate_tlearner,
    _estimate_cate_xlearner, _estimate_ltv_churn, _estimate_ltv_proxy,
    _estimate_slearner, _wellspec_value_model, seed_from,
)
from core.io import add_common_args, resolve_run_size, save_json
from core.policy import (
    DEFAULT_BUDGET, THETA_GRID, calibrate_theta_all, f_churn_vec, target_topk,
)
from core.stats import holm_bonferroni, paired_test_raw
from study_learners import load_or_tune, welfare_at_kappa

KAPPA = DEFAULT_BUDGET
ADV_KAPPAS = (0.30, 0.10)
ORACLE_MC = 400            # noise draws per unit for the feasible oracle (as study_main)
ARMS = ('default', 'tuned')
ORACLES = ('Oracle hindsight', 'Oracle feasible')
ADV_DROP_COLS = [0, 1, 2]   # bonus_history (index 3) dropped from the CATE model
LOGIN_DROP_COLS = [0, 1, 3]  # login_freq (index 2) dropped
MISSPEC_DROP_COLS = {2: LOGIN_DROP_COLS, 3: ADV_DROP_COLS}
ALL_COLS = [0, 1, 2, 3]


def _arm(name, pset):
    return name if pset == 'default' else f'{name} (tuned)'


# ---------------------------------------------------------------------------
# Shared per-replication machinery
# ---------------------------------------------------------------------------
def _setup(dgp_func, seed, n_players):
    """Simulate one replication and split it; returns the test-split truth."""
    data = dgp_func(n_players, np.random.RandomState(seed))
    idx = np.arange(len(data['treatment']))
    tr, te = train_test_split(idx, test_size=0.30, random_state=seed)
    X, W, Y = data['features'], data['treatment'], data['y_obs']
    return SimpleNamespace(
        data=data, seed=seed, tr=tr, te=te,
        Xtr=X[tr], Xte=X[te], Wtr=W[tr], Ytr=Y[tr],
        tau=data['true_tau'][te], ltv=data['true_ltv'][te])


def _tau_arms(rep, tuned, cols=ALL_COLS):
    """T-learner tau_hat per arm (default params, tuned params), on the given columns."""
    return {pset: _estimate_cate_tlearner(rep.Xtr[:, cols], rep.Xte[:, cols],
                                          rep.Wtr, rep.Ytr,
                                          tuned['T'] if pset == 'tuned' else None)
            for pset in ARMS}


def _est_func(params):
    """est_func for calibrate_theta_all: a T-learner with `params` (None: the legacy
    estimate_components fit) and the legacy value and churn models."""
    def est(data, train_idx, test_idx, cate_method='tlearner', *, seed, stream=LTV_STREAM):
        X, W, Y = data['features'], data['treatment'], data['y_obs']
        tau = _estimate_cate_tlearner(X[train_idx], X[test_idx], W[train_idx],
                                      Y[train_idx], params)
        ltv, churn = _estimate_ltv_churn(X[train_idx], X[test_idx], W[train_idx],
                                         Y[train_idx], seed, stream)
        return tau, ltv, churn
    return est


def _calibrate(rep, kappa, pset, tuned, objectives=('aipw',)):
    return calibrate_theta_all(rep.data, rep.tr, rep.seed, kappa, objectives,
                               est_func=_est_func(tuned['T'] if pset == 'tuned' else None))


def _oracle_cm(rep, dgp_func):
    """Feasible-oracle conditional means on the test features (once per replication)."""
    name, extra = dgp_spec(dgp_func)
    if name == 'corr_gamma':
        # the DGP standardises tau_base on the full generating sample; X is observable
        extra = dict(extra, tau_base_stats=tau_base_stats(
            dict(zip(FEATURE_NAMES, rep.data['features'].T))))
    feats = {nm: rep.Xte[:, i] for i, nm in enumerate(FEATURE_NAMES)}
    return conditional_means(name, feats, n_mc=ORACLE_MC,
                             seed=seed_from(rep.seed, ORACLE_STREAM), **extra)


def _oracle_welfare(rep, cm, kappa, tau_eval=None):
    """{'Oracle hindsight', 'Oracle feasible'} welfare at kappa (g = 1); tau_eval
    replaces the true tau in the welfare sum (effective_welfare)."""
    tau_eval = rep.tau if tau_eval is None else tau_eval
    return {
        'Oracle hindsight': welfare_at_kappa(rep.tau * rep.ltv, kappa, tau_eval, rep.ltv),
        'Oracle feasible': welfare_at_kappa(cm['tau_bar'] * cm['ltv_bar'], kappa,
                                            tau_eval, rep.ltv)}


def _map(fn, reps, n_jobs):
    return Parallel(n_jobs=n_jobs)(delayed(fn)(SEED_BASE + 7 * r) for r in range(reps))


def _stack(rows):
    """[{name: value}, ...] over replications -> {name: array}."""
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def _w_summary(v):
    v = np.asarray(v, dtype=float)
    n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else float('nan')
    half = float(scipy_stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n) if n > 1 else float('nan')
    return {'mean': float(v.mean()), 'sd': sd, 'ci95': [float(v.mean()) - half,
                                                         float(v.mean()) + half]}


def _cell(welfare, pairs):
    """Per-cell JSON: welfare summaries and a paired test of a - b for each pair."""
    return {
        'welfare': {s: _w_summary(v) for s, v in welfare.items()},
        'comparisons': {f'{a} vs {b}': {'a': a, 'b': b,
                                        **paired_test_raw(welfare[a], welfare[b])}
                        for a, b in pairs}}


def _apply_holm(cells):
    """Add p_holm to every non-degenerate comparison, pooled over all cells; returns
    the family size.  A comparison flagged degenerate (zero by construction) is kept
    but excluded: its p_holm is None."""
    all_c = [c for cell in cells.values() for c in cell['comparisons'].values()]
    comps = [c for c in all_c if not c.get('degenerate')]
    for c in all_c:
        if c.get('degenerate'):
            c['p_holm'] = None
    if comps:
        adj = holm_bonferroni(np.array([c['p_value'] for c in comps]))
        for c, a in zip(comps, adj):
            c['p_holm'] = float(a)
    return len(comps)


def _finish(cells, family, **meta):
    m = _apply_holm(cells)
    return {'cells': cells, 'holm_family': family, 'holm_family_size': m, **meta}


def _ltv_advantage(cell):
    """Percent gain of CATE x LTV over CATE Only per arm, from the cell means."""
    w = cell['welfare']
    return {p: 100 * (w[_arm('CATE x LTV', p)]['mean'] - w[_arm('CATE Only', p)]['mean'])
            / w[_arm('CATE Only', p)]['mean'] for p in ARMS}


def _calibration_summary(idxs):
    c = Counter(idxs)
    return {'identity_selected': int(c.get(0, 0)), 'n_runs': len(idxs),
            'identity_rate': c.get(0, 0) / len(idxs),
            'selected_config_counts': {str(k): v for k, v in c.most_common()}}


# ---------------------------------------------------------------------------
# slearner_scenarios, asym, corr_sweep, heavy_tail: the generic DGP comparison
# ---------------------------------------------------------------------------
def _dgp_rep(seed, *, dgp_func, n_players, tuned, kappa, calibrate, cols=ALL_COLS):
    rep = _setup(dgp_func, seed, n_players)
    taus = _tau_arms(rep, tuned, cols)
    ltv_hat, churn_hat = _estimate_ltv_churn(rep.Xtr, rep.Xte, rep.Wtr, rep.Ytr, seed)
    wel = lambda s: welfare_at_kappa(s, kappa, rep.tau, rep.ltv)
    w, cal_idx = {}, {}
    for pset, tau_hat in taus.items():
        w[_arm('CATE Only', pset)] = wel(tau_hat)
        w[_arm('CATE x LTV', pset)] = wel(tau_hat * ltv_hat)
        if calibrate:
            w[_arm('Full Framework (Est.)', pset)] = wel(
                tau_hat * ltv_hat * f_churn_vec(churn_hat))
            cfg = _calibrate(rep, kappa, pset, tuned)['aipw']
            cal_idx[pset] = THETA_GRID.index(cfg)
            w[_arm('Full Framework (Cal.)', pset)] = wel(
                tau_hat * ltv_hat * f_churn_vec(churn_hat, **cfg))
    w['S-Learner'] = wel(_estimate_slearner(rep.Xtr, rep.Xte, rep.Wtr, rep.Ytr, ltv_hat))
    w.update(_oracle_welfare(rep, _oracle_cm(rep, dgp_func), kappa))
    feats = rep.data
    corr = float(np.corrcoef(feats['true_tau'], feats['true_ltv'])[0, 1])
    return {'w': w, 'cal_idx': cal_idx, 'corr_tau_V': corr}


def _dgp_pairs(calibrate):
    pairs = [(_arm('CATE x LTV', p), _arm('CATE Only', p)) for p in ARMS]
    pairs += [('S-Learner', 'CATE x LTV'), ('CATE x LTV (tuned)', 'CATE x LTV')]
    pairs += [('Oracle feasible', _arm('CATE x LTV', p)) for p in ARMS]
    if calibrate:
        pairs += [(_arm('Full Framework (Cal.)', p), _arm('CATE x LTV', p)) for p in ARMS]
    return pairs


def _dgp_scenario(dgp_cells, family, reps, n_players, tuned, n_jobs, *,
                  kappa=KAPPA, calibrate=False):
    cells = {}
    for name, spec in dgp_cells.items():
        dgp_func, cols = spec if isinstance(spec, tuple) else (spec, ALL_COLS)
        rows = _map(partial(_dgp_rep, dgp_func=dgp_func, n_players=n_players,
                            tuned=tuned, kappa=kappa, calibrate=calibrate, cols=cols),
                    reps, n_jobs)
        cell = _cell(_stack([r['w'] for r in rows]), _dgp_pairs(calibrate))
        cell['kappa'] = kappa
        cell['corr_tau_V'] = float(np.mean([r['corr_tau_V'] for r in rows]))
        cell['ltv_advantage_pct'] = _ltv_advantage(cell)
        if calibrate:
            cell['calibration'] = {p: _calibration_summary([r['cal_idx'][p] for r in rows])
                                   for p in ARMS}
        cells[name] = cell
    return _finish(cells, family, kappa=kappa)


def run_slearner_scenarios(reps, n_players, tuned, n_jobs=1):
    out = _dgp_scenario(
        {'main': DGPS['main'], 'null': DGPS['null'], 'correlated': DGPS['correlated'],
         'misspec': (DGPS['main'], LOGIN_DROP_COLS)},
        'all comparisons in the main, null, correlated and misspec cells',
        reps, n_players, tuned, n_jobs)
    out['misspec_note'] = ('misspec cell: main DGP, the T-learner CATE model (both arms) drops '
                           'login_freq while the S-learner keeps all four features, as in the '
                           'paper; S-Learner vs CATE x LTV there is the paired test of that '
                           'contrast.  The main cell is the full-feature comparison.')
    return out


def run_asym(reps, n_players, tuned, n_jobs=1):
    return _dgp_scenario({'asym': DGPS['asym']},
                         'all comparisons on the asymmetric DGP',
                         reps, n_players, tuned, n_jobs, calibrate=True)


def run_corr_sweep(reps, n_players, tuned, n_jobs=1):
    cells = {label: make_corr_dgp(g)
             for label, g in (('negative', -1.0), ('zero', 0.0), ('positive', 1.0))}
    out = _dgp_scenario(cells, 'all comparisons across the three gamma cells',
                        reps, n_players, tuned, n_jobs)
    out['gamma'] = {'negative': -1.0, 'zero': 0.0, 'positive': 1.0}
    return out


def run_heavy_tail(reps, n_players, tuned, n_jobs=1):
    return _dgp_scenario({'heavy_tail': DGPS['heavytail']},
                         'all comparisons on the heavy-tailed DGP',
                         reps, n_players, tuned, n_jobs)


# ---------------------------------------------------------------------------
# effective_welfare
# ---------------------------------------------------------------------------
def _effective_rep(seed, *, n_players, tuned, kappa):
    dgp_func = DGPS['main']
    rep = _setup(dgp_func, seed, n_players)
    d = rep.data
    eff_tau = (d['y1_prob'] - d['y0_prob'])[rep.te]
    taus = _tau_arms(rep, tuned)
    ltv_hat, churn_hat = _estimate_ltv_churn(rep.Xtr, rep.Xte, rep.Wtr, rep.Ytr, seed)
    cm = _oracle_cm(rep, dgp_func)
    scores = {}
    for pset, tau_hat in taus.items():
        scores[_arm('CATE Only', pset)] = tau_hat
        scores[_arm('CATE x LTV', pset)] = tau_hat * ltv_hat
        cfg = _calibrate(rep, kappa, pset, tuned)['aipw']
        scores[_arm('Full Framework (Cal.)', pset)] = (
            tau_hat * ltv_hat * f_churn_vec(churn_hat, **cfg))
    scores['S-Learner'] = _estimate_slearner(rep.Xtr, rep.Xte, rep.Wtr, rep.Ytr, ltv_hat)
    scores['Oracle hindsight'] = rep.tau * rep.ltv
    scores['Oracle feasible'] = cm['tau_bar'] * cm['ltv_bar']
    w = {}
    for s, sc in scores.items():
        nominal = welfare_at_kappa(sc, kappa, rep.tau, rep.ltv)
        effective = welfare_at_kappa(sc, kappa, eff_tau, rep.ltv)
        w[s], w[f'{s} [effective]'] = nominal, effective
        w[f'{s} [overstatement]'] = nominal - effective
    return w


def run_effective_welfare(reps, n_players, tuned, n_jobs=1):
    w = _stack(_map(partial(_effective_rep, n_players=n_players, tuned=tuned,
                            kappa=KAPPA), reps, n_jobs))
    strategies = [k for k in w if '[' not in k]
    pairs = [(s, f'{s} [effective]') for s in strategies]
    pairs += [(f'{s} [overstatement]', 'CATE Only [overstatement]')
              for s in strategies if s != 'CATE Only']
    cells = {'main': _cell(w, pairs)}
    over = {s: {'nominal': float(w[s].mean()), 'effective': float(w[f'{s} [effective]'].mean()),
                'pct_overstate': 100 * float(w[f'{s} [overstatement]'].mean()
                                              / w[f'{s} [effective]'].mean())}
            for s in strategies}
    return _finish(cells, 'nominal-vs-effective per strategy and each strategy\'s '
                          'overstatement vs CATE Only', kappa=KAPPA, overstatement=over)


# ---------------------------------------------------------------------------
# component_update
# ---------------------------------------------------------------------------
def _component_rep(seed, *, n_players, tuned, kappa):
    dgp_func = DGPS['main']
    rep = _setup(dgp_func, seed, n_players)
    taus = _tau_arms(rep, tuned)
    v_mis = _estimate_ltv_proxy(rep.Xtr, rep.Xte, seed)
    v_well = _wellspec_value_model(rep.Xtr, rep.Xte, seed)
    wel = lambda s: welfare_at_kappa(s, kappa, rep.tau, rep.ltv)
    w = {}
    for pset, tau_hat in taus.items():
        w[_arm('CATE x LTV, misspecified V', pset)] = wel(tau_hat * v_mis)
        w[_arm('CATE x LTV, well-specified V', pset)] = wel(tau_hat * v_well)
    w.update(_oracle_welfare(rep, _oracle_cm(rep, dgp_func), kappa))
    return w


def run_component_update(reps, n_players, tuned, n_jobs=1):
    w = _stack(_map(partial(_component_rep, n_players=n_players, tuned=tuned,
                            kappa=KAPPA), reps, n_jobs))
    pairs = [(_arm('CATE x LTV, well-specified V', p), _arm('CATE x LTV, misspecified V', p))
             for p in ARMS]
    cell = _cell(w, pairs)
    recovery = {}
    for p in ARMS:
        a = cell['welfare'][_arm('CATE x LTV, misspecified V', p)]['mean']
        b = cell['welfare'][_arm('CATE x LTV, well-specified V', p)]['mean']
        recovery[p] = {'recovery_abs': b - a, 'recovery_pct': 100 * (b - a) / a}
    return _finish({'main': cell}, 'well-specified vs misspecified V, both arms',
                   kappa=KAPPA, recovery=recovery)


# ---------------------------------------------------------------------------
# obs_diagnostic
# ---------------------------------------------------------------------------
def _misrank_terms(tau_hat, v_hat, tau_ref, v_ref, k):
    """(mean |tau_ref - tau_hat| |V_hat|, mean |tau_hat| |V_ref - V_hat|) over the
    misranked set (top-k by deployed XOR by reference), or None if it is empty."""
    pi_dep = set(target_topk(tau_hat * v_hat, k).tolist())
    pi_ref = set(target_topk(tau_ref * v_ref, k).tolist())
    m = np.array(sorted(pi_dep.symmetric_difference(pi_ref)), dtype=int)
    if len(m) == 0:
        return None
    return (float((np.abs(tau_ref[m] - tau_hat[m]) * np.abs(v_hat[m])).mean()),
            float((np.abs(tau_hat[m]) * np.abs(v_ref[m] - v_hat[m])).mean()))


def _obs_rep(seed, *, n_players, tuned, kappa):
    dgp_func = DGPS['main']
    rep = _setup(dgp_func, seed, n_players)
    k = int(len(rep.te) * kappa)
    cm = _oracle_cm(rep, dgp_func)
    v_hat = _estimate_ltv_proxy(rep.Xtr, rep.Xte, seed)
    # observable reference: X-learner tau (all features), richer value model on the same
    # observable proxy target
    tau_ref = _estimate_cate_xlearner(rep.Xtr, rep.Xte, rep.Wtr, rep.Ytr)
    rng_v = np.random.RandomState(seed_from(seed, VALUE_STREAM_V43))
    vproxy = (12 * rep.Xtr[:, 1] ** 0.9 + 35 * rep.Xtr[:, 2]
              + rng_v.normal(0, 200, len(rep.Xtr)))
    vref = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     random_state=1).fit(rep.Xtr, vproxy)
    v_ref = vref.predict(rep.Xte).clip(50, 8000)
    refs = {'plug-in': (tau_ref, v_ref), 'Oracle hindsight': (rep.tau, rep.ltv),
            'Oracle feasible': (cm['tau_bar'], cm['ltv_bar'])}
    out = {}
    for cell, cols in (('wellspec', ALL_COLS), ('misspec', LOGIN_DROP_COLS)):
        for pset, tau_hat in _tau_arms(rep, tuned, cols).items():
            for ref, (t, v) in refs.items():
                out[(cell, pset, ref)] = _misrank_terms(tau_hat, v_hat, t, v, k)
    return out


def run_obs_diagnostic(reps, n_players, tuned, n_jobs=1):
    rows = _map(partial(_obs_rep, n_players=n_players, tuned=tuned, kappa=KAPPA),
                reps, n_jobs)
    refs = ('plug-in', 'Oracle hindsight', 'Oracle feasible')
    cells = {c: {'diagnostic': {p: {} for p in ARMS}} for c in ('wellspec', 'misspec')}
    shares = {}
    for key in rows[0]:
        cell, pset, ref = key
        terms = [r[key] for r in rows]
        used = [t for t in terms if t is not None]
        a = float(np.mean([t[0] for t in used])) if used else float('nan')
        b = float(np.mean([t[1] for t in used])) if used else float('nan')
        cells[cell]['diagnostic'][pset][ref] = {
            'cate_term': a, 'value_term': b,
            'cate_share_pct': 100 * a / (a + b) if used and a + b > 0 else float('nan'),
            'n_reps_used': len(used)}
        shares[key] = [None if t is None else 100 * t[0] / (t[0] + t[1]) for t in terms]
    comps = {}
    for pset in ARMS:
        for ref in refs:
            mis, well = shares[('misspec', pset, ref)], shares[('wellspec', pset, ref)]
            both = [(m, w) for m, w in zip(mis, well) if m is not None and w is not None]
            label = f'CATE share, misspec vs wellspec [{_arm("T", pset)}, {ref}]'
            comps[label] = {'n_pairs': len(both),
                            **(paired_test_raw([m for m, _ in both], [w for _, w in both])
                               if len(both) > 1 else
                               {'mean_diff': float('nan'), 'se_diff': float('nan'),
                                't_stat': float('nan'), 'p_value': 1.0,
                                'cohens_d': float('nan'), 'ci_95': [float('nan')] * 2})}
    m = _apply_holm({'all': {'comparisons': comps}})
    return {'cells': cells, 'comparisons': comps, 'kappa': KAPPA,
            'holm_family': 'per-replication CATE share, misspec vs wellspec, over the '
                           '2 arms x 3 references',
            'holm_family_size': m}


# ---------------------------------------------------------------------------
# aipw_calibration
# ---------------------------------------------------------------------------
def _aipw_rep(seed, *, n_players, tuned, kappa):
    dgp_func = DGPS['main']
    rep = _setup(dgp_func, seed, n_players)
    ltv_hat, churn_hat = _estimate_ltv_churn(rep.Xtr, rep.Xte, rep.Wtr, rep.Ytr, seed)
    wel = lambda s: welfare_at_kappa(s, kappa, rep.tau, rep.ltv)
    w, idx = {}, {}
    for pset, tau_hat in _tau_arms(rep, tuned).items():
        base = tau_hat * ltv_hat
        w[_arm('Identity (g=1)', pset)] = wel(base)
        cals = _calibrate(rep, kappa, pset, tuned, ('aipw', 'plugin'))
        for obj, name in (('aipw', 'Calibrated (aipw)'), ('plugin', 'Calibrated (plug-in)')):
            idx[(pset, obj)] = THETA_GRID.index(cals[obj])
            w[_arm(name, pset)] = wel(base * f_churn_vec(churn_hat, **cals[obj]))
    w.update(_oracle_welfare(rep, _oracle_cm(rep, dgp_func), kappa))
    return {'w': w, 'idx': idx}


def run_aipw_calibration(reps, n_players, tuned, n_jobs=1):
    rows = _map(partial(_aipw_rep, n_players=n_players, tuned=tuned, kappa=KAPPA),
                reps, n_jobs)
    pairs = [(_arm(f'Calibrated ({o})', p), _arm('Identity (g=1)', p))
             for p in ARMS for o in ('aipw', 'plug-in')]
    cell = _cell(_stack([r['w'] for r in rows]), pairs)
    for c in cell['comparisons'].values():
        # the plug-in objective always selects identity, so its difference is exactly 0
        # (p = 1) by construction: kept in the output, excluded from the Holm family
        c['degenerate'] = c['a'].startswith('Calibrated (plug-in)')
    cell['selection'] = {p: {o: _calibration_summary([r['idx'][(p, o)] for r in rows])
                             for o in ('aipw', 'plugin')} for p in ARMS}
    cell['kappa'] = KAPPA
    out = _finish({'main': cell}, 'AIPW-calibrated vs identity, both arms (the degenerate plug-in comparisons are excluded)',
                  kappa=KAPPA)
    out['note'] = ('config index 0 is the near-identity g~1; index 3 is default theta0.  '
                   'A comparison mean_diff is the true test welfare of the calibrated rule '
                   'minus the g=1 (CATE x LTV) rule.  The plug-in objective selects '
                   'identity by construction (difference exactly 0, p = 1; flagged degenerate and excluded from the Holm family).  Proposition 3 '
                   'implies no g improves on identity at the population optimum (true tau '
                   'and V); with estimated components that need not hold, so a negative '
                   'or zero mean is an empirical observation, not a consequence of the '
                   'proposition.')
    return out


# ---------------------------------------------------------------------------
# adversarial
# ---------------------------------------------------------------------------
def _adv_rep(seed, *, n_players, tuned, kappas):
    dgp_func = generate_adv
    rep = _setup(dgp_func, seed, n_players)
    ltv_hat = _estimate_ltv_proxy(rep.Xtr, rep.Xte, seed)
    drop = _tau_arms(rep, tuned, ADV_DROP_COLS)
    full = _tau_arms(rep, tuned, ALL_COLS)
    s_learner = _estimate_slearner(rep.Xtr, rep.Xte, rep.Wtr, rep.Ytr, ltv_hat)
    cm = _oracle_cm(rep, dgp_func)
    out = {}
    for kp in kappas:
        wel = lambda s: welfare_at_kappa(s, kp, rep.tau, rep.ltv)
        w = {}
        for pset in ARMS:
            w[_arm('CATE Only', pset)] = wel(drop[pset])
            w[_arm('CATE x LTV', pset)] = wel(drop[pset] * ltv_hat)
            w[_arm('CATE Only (well-specified)', pset)] = wel(full[pset])
            w[_arm('CATE x LTV (well-specified)', pset)] = wel(full[pset] * ltv_hat)
        w['S-Learner'] = wel(s_learner)      # sees bonus_history
        w.update(_oracle_welfare(rep, cm, kp))
        out[f'kappa_{kp:.2f}'] = w
    return out


def run_adversarial(reps, n_players, tuned, n_jobs=1):
    rows = _map(partial(_adv_rep, n_players=n_players, tuned=tuned, kappas=ADV_KAPPAS),
                reps, n_jobs)
    pairs = []
    for p in ARMS:
        pairs += [(_arm('CATE x LTV', p), _arm('CATE Only', p)),
                  (_arm('CATE x LTV (well-specified)', p),
                   _arm('CATE Only (well-specified)', p)),
                  ('Oracle feasible', _arm('CATE x LTV', p))]
    pairs += [('S-Learner', 'CATE x LTV'), ('CATE x LTV (tuned)', 'CATE x LTV')]
    cells = {}
    for name in rows[0]:
        cell = _cell(_stack([r[name] for r in rows]), pairs)
        cell['kappa'] = float(name.split('_')[1])
        cells[name] = cell
    out = _finish(cells, 'all comparisons across both kappa cells')
    out['interpretation'] = (
        'If CATE x LTV minus CATE Only is <= 0 under the dropout (value weighting does NOT '
        'beat pure CATE), the value model failed to rescue the dropped signal, bounding '
        'the favourable login-drop result in the main misspecification table.')
    return out


# ---------------------------------------------------------------------------
# like_for_like
# ---------------------------------------------------------------------------
def _like_rep(seed, *, dgp_func, drop_idx, n_players, tuned, kappa):
    rep = _setup(dgp_func, seed, n_players)
    keep = MISSPEC_DROP_COLS[drop_idx]
    ltv_hat = _estimate_ltv_proxy(rep.Xtr, rep.Xte, seed)
    wel = lambda s: welfare_at_kappa(s, kappa, rep.tau, rep.ltv)
    w = {'S-Learner (all features)': wel(_estimate_slearner(
             rep.Xtr, rep.Xte, rep.Wtr, rep.Ytr, ltv_hat)),
         'S-Learner (feature dropped)': wel(_estimate_slearner(
             rep.Xtr[:, keep], rep.Xte[:, keep], rep.Wtr, rep.Ytr, ltv_hat))}
    full, dropped = _tau_arms(rep, tuned, ALL_COLS), _tau_arms(rep, tuned, keep)
    for pset in ARMS:
        name = (lambda t: f'CATE Only ({t})' if pset == 'default'
                else f'CATE Only (tuned, {t})')
        w[name('all features')] = wel(full[pset])
        w[name('feature dropped')] = wel(dropped[pset])
    w.update(_oracle_welfare(rep, _oracle_cm(rep, dgp_func), kappa))
    return w


def run_like_for_like(reps, n_players, tuned, n_jobs=1):
    pairs = [('S-Learner (all features)', 'S-Learner (feature dropped)'),
             ('CATE Only (all features)', 'CATE Only (feature dropped)'),
             ('CATE Only (tuned, all features)', 'CATE Only (tuned, feature dropped)')]
    cells = {}
    for name, dgp_func, drop_idx in (('login_drop_main', DGPS['main'], 2),
                                     ('bonus_drop_adversarial', generate_adv, 3)):
        w = _stack(_map(partial(_like_rep, dgp_func=dgp_func, drop_idx=drop_idx,
                                n_players=n_players, tuned=tuned, kappa=KAPPA),
                        reps, n_jobs))
        cell = _cell(w, pairs)
        # comparison keys read 'X (all features) vs X (feature dropped)'; a - b = degradation
        cell['comparisons'] = {
            ('S-Learner: all features vs dropped' if c['a'].startswith('S-Learner') else k): c
            for k, c in cell['comparisons'].items()}
        cell['dropped_feature_index'] = drop_idx
        cell['dropped_feature'] = FEATURE_NAMES[drop_idx]
        cell['kappa'] = KAPPA
        cells[name] = cell
    return _finish(cells, 'all-features vs dropped, S-learner and both T arms, both cells',
                   kappa=KAPPA)


SCENARIOS = {
    'slearner_scenarios': run_slearner_scenarios,
    'asym': run_asym,
    'effective_welfare': run_effective_welfare,
    'component_update': run_component_update,
    'corr_sweep': run_corr_sweep,
    'heavy_tail': run_heavy_tail,
    'obs_diagnostic': run_obs_diagnostic,
    'aipw_calibration': run_aipw_calibration,
    'adversarial': run_adversarial,
    'like_for_like': run_like_for_like,
}


def run_all(reps, n_players, tuned, n_jobs=1):
    """Run every scenario; returns ({name: result}, {name: wall seconds})."""
    out, timing = {}, {}
    for name, fn in SCENARIOS.items():
        t0 = time.perf_counter()
        out[name] = fn(reps, n_players, tuned, n_jobs=n_jobs)
        timing[name] = time.perf_counter() - t0
        print(f"  {name:20} {timing[name]:7.1f}s", flush=True)
    return out, timing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_common_args(parser)
    parser.add_argument('--retune', action='store_true',
                        help='rerun the tuning pilot even if the frozen file exists')
    parser.add_argument('--jobs', type=int, default=1, help='parallel worker processes')
    args = parser.parse_args(argv)
    reps, n_players = resolve_run_size(args, N_RUNS, N_PLAYERS)
    print(f"Robustness scenarios | reps={reps}, n={n_players}")
    tuned = load_or_tune(n_players=n_players, retune=args.retune, n_jobs=args.jobs)
    out, timing = run_all(reps, n_players, tuned, n_jobs=args.jobs)
    out['_meta'] = {'n_runs': reps, 'n_players': n_players, 'seeds_base': SEED_BASE,
                    'kappa': KAPPA, 'tuned_T_params': tuned['T'], 'tuned_on': 'main',
                    'seconds_by_scenario': timing,
                    'seconds_per_rep': sum(timing.values()) / reps}
    name = 'robustness_smoke.json' if args.smoke else 'robustness.json'
    print(f"Saved {save_json(out, name)}")


if __name__ == '__main__':
    main()
