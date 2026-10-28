"""Main simulation study: risk-modulated value-weighted targeting.

Scenarios (all at seeds 42+7r, N_PLAYERS x N_RUNS unless overridden):
main DGP with the T-learner (strategies, frontier, inverted-g misspecification,
calibration summary, Holm-corrected paired tests), X-learner, CATE misspecification
(with a TOST equivalence test, kept because it supports the equivalence claim),
null and correlated DGPs, and the budget sensitivity sweep.  The hindsight and
feasible tau*V oracle tables (oracle_main, oracle_budget) come from the same runs.
Descriptive SDs across replications use ddof=1.

Writes outputs/main.json (main_smoke.json with --smoke).
Replications are independent (per-rep seeds 42 + 7r), so --jobs J runs them in a joblib
worker pool with results in replication order: output is identical for every J.
main_results and budget_sensitivity are stored unrounded (rounding happens when tables are
formatted).  'efficiency' holds per-replication paired ratios against the feasible oracle.

Usage:  python study_main.py [--reps R] [--n-players N] [--smoke] [--verify-moments] [--jobs J]
"""

import argparse
from collections import Counter

import numpy as np
from joblib import Parallel, delayed
from scipy import stats as scipy_stats
from sklearn.model_selection import train_test_split

from core.dgp import (
    DGP_SPEC,
    FEATURE_NAMES,
    ORACLE_MC,
    POP_MEAN,
    POP_SD,
    TREATMENT_PROB,
    conditional_means,
    dgp_spec,
    tau_base_stats,
)
from core.dgp import (
    generate_correlated as generate_population_correlated,
)
from core.dgp import (
    generate_main as generate_population,
)
from core.dgp import (
    generate_null as generate_population_null,
)
from core.estimators import (
    ORACLE_STREAM,
    _estimate_slearner,
    estimate_components,
    estimate_components_misspec,
    seed_from,
)
from core.io import add_common_args, resolve_run_size, save_json
from core.policy import (
    DEFAULT_BUDGET,
    THETA_GRID,
    calibrate_theta_all,
    compute_mean_churn_moderation,
    compute_risk_exposure,
    compute_welfare,
    f_churn_vec,
    target_topk,
)
from core.stats import (
    holm_bonferroni,
    paired_test_raw,
    ratio_entry,
    t_mean_ci,
    tost_paired,
)

# --- Configuration ---
N_PLAYERS = 10000
N_RUNS = 200
BUDGET_FRACTIONS = [0.10, 0.20, 0.30, 0.50]
SEED_BASE = 42
ORACLE_STRATEGIES = ['Oracle hindsight (g=1)', 'Oracle feasible (g=1)',
                     'Oracle feasible (theta0)']

# --- Main simulation loop ---

def run_single_simulation(seed, budget_frac=DEFAULT_BUDGET, dgp_func=None,
                          est_func=None, cate_method='tlearner', n_players=None):
    if n_players is None:
        n_players = N_PLAYERS
    if dgp_func is None:
        dgp_func = generate_population
    if est_func is None:
        est_func = estimate_components
    rng = np.random.RandomState(seed)
    data = dgp_func(n_players, rng)

    n = len(data['treatment'])
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.30, random_state=seed)

    k = int(len(test_idx) * budget_frac)
    true_tau = data['true_tau'][test_idx]
    true_ltv = data['true_ltv'][test_idx]
    true_churn = data['true_churn'][test_idx]
    y0 = data['y0'][test_idx]
    y1 = data['y1'][test_idx]

    tau_hat, ltv_hat, churn_hat = est_func(data, train_idx, test_idx,
                                           cate_method=cate_method, seed=seed)

    g_hat = f_churn_vec(churn_hat)
    g_oracle = f_churn_vec(true_churn)

    # Calibrate at the evaluation budget (budget_frac), AIPW objective; the
    # plug-in objective is kept only to exhibit the top-k tautology (g=1).
    cals = calibrate_theta_all(data, train_idx, seed, budget_frac,
                               est_func=est_func, cate_method=cate_method)
    cal_config = cals['aipw']
    g_cal = f_churn_vec(churn_hat, **cal_config)
    cal_plugin = cals['plugin']
    g_cal_plugin = f_churn_vec(churn_hat, **cal_plugin)

    strategies = {}
    strategies['Random'] = rng.uniform(0, 1, len(test_idx))
    strategies['CATE Only'] = tau_hat
    strategies['CATE x LTV'] = tau_hat * ltv_hat
    strategies['Full Framework (Est.)'] = tau_hat * ltv_hat * g_hat
    strategies['Full Framework (Cal.)'] = tau_hat * ltv_hat * g_cal
    strategies['Full Framework (Cal., plug-in)'] = tau_hat * ltv_hat * g_cal_plugin
    strategies['Oracle hindsight (theta0)'] = (data['true_tau'][test_idx]
                                              * data['true_ltv'][test_idx]
                                              * g_oracle)

    # Oracles. Hindsight ranks on realised tau, V (includes idiosyncratic noise,
    # unreachable by any X-measurable rule); feasible ranks on E[tau|X] E[V|X].
    # Welfare of all of them is evaluated on realised true_tau, true_ltv.
    strategies['Oracle hindsight (g=1)'] = true_tau * true_ltv
    dgp_name, extra = dgp_spec(dgp_func)   # raises ValueError if unknown
    if dgp_name == 'corr_gamma':
        # DGP standardises tau_base on the full generating sample; X is observable.
        extra = dict(extra, tau_base_moments=tau_base_stats(
            dict(zip(FEATURE_NAMES, data['features'].T))))
    feats = {nm: data['features'][test_idx, i]
             for i, nm in enumerate(FEATURE_NAMES)}
    cm = conditional_means(dgp_name, feats, n_mc=ORACLE_MC,
                           seed=seed_from(seed, ORACLE_STREAM), **extra)
    tv_bar = cm['tau_bar'] * cm['ltv_bar']
    strategies['Oracle feasible (g=1)'] = tv_bar
    strategies['Oracle feasible (theta0)'] = tv_bar * f_churn_vec(cm['churn_bar'])

    # S-Learner baseline [C3]
    X_train = data['features'][train_idx]
    X_test = data['features'][test_idx]
    W_train = data['treatment'][train_idx]
    Y_train = data['y_obs'][train_idx]
    strategies['S-Learner'] = _estimate_slearner(
        X_train, X_test, W_train, Y_train, ltv_hat)

    results = {}
    for name, scores in strategies.items():
        targeted = target_topk(scores, k)
        ate = float(np.mean(y1[targeted] - y0[targeted]) * 100)
        welfare = compute_welfare(targeted, true_tau, true_ltv)
        risk = compute_risk_exposure(targeted, true_churn)
        mod_factor = compute_mean_churn_moderation(targeted, true_churn)
        causally_retained = targeted[(y1[targeted] - y0[targeted]) == 1]
        mean_ltv_cr = (float(np.mean(true_ltv[causally_retained]))
                       if len(causally_retained) > 0 else 0.0)
        results[name] = {
            'ate': ate, 'welfare': welfare, 'risk_exposure': risk,
            'mean_churn_mod': mod_factor, 'mean_ltv_cr': mean_ltv_cr,
        }

    # Frontier (using estimated components, varying theta)
    frontier_configs = [
        ('No reg (g=1)', None),
        ('Wide (0.03/0.50/0.70)', {'pL': 0.03, 'pML': 0.50, 'pMH': 0.70}),
        ('Default (0.05/0.40/0.60)', {}),
        ('Narrow (0.10/0.30/0.50)', {'pL': 0.10, 'pML': 0.30, 'pMH': 0.50}),
        ('Aggressive (0.15/0.35/0.50/phi=0.05)',
         {'pL': 0.15, 'pML': 0.35, 'pMH': 0.50, 'phi': 0.05}),
        ('Calibrated', None),
    ]
    frontier = []
    for cfg_name, kwargs in frontier_configs:
        if cfg_name == 'No reg (g=1)':
            scores = tau_hat * ltv_hat
        elif cfg_name == 'Calibrated':
            scores = tau_hat * ltv_hat * g_cal
        else:
            g_cfg = f_churn_vec(churn_hat, **kwargs)
            scores = tau_hat * ltv_hat * g_cfg
        targeted = target_topk(scores, k)
        frontier.append({
            'config': cfg_name,
            'welfare': compute_welfare(targeted, true_tau, true_ltv),
            'risk_exposure': compute_risk_exposure(targeted, true_churn),
            'mean_churn_mod': compute_mean_churn_moderation(targeted, true_churn),
        })

    # Misspecification: inverted g
    g_inv = f_churn_vec(1.0 - churn_hat)
    targeted_inv = target_topk(tau_hat * ltv_hat * g_inv, k)
    misspec = {
        'welfare': compute_welfare(targeted_inv, true_tau, true_ltv),
        'risk_exposure': compute_risk_exposure(targeted_inv, true_churn),
    }

    return results, frontier, misspec, cal_config


def _sd(values, ndigits=None):
    """Descriptive SD across replications (sample SD, ddof=1); None for fewer than 2."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    sd = float(np.std(values, ddof=1))
    return sd if ndigits is None else round(sd, ndigits)


def _oracle_table(runs_by_name, reps, n):
    """Oracle benchmarks at the default budget (unrounded).

    Hindsight ranks on realised tau, V (includes idiosyncratic noise, so no
    X-measurable rule reaches it); feasible ranks on E[tau|X] E[V|X].  Both are
    oracle inputs, not estimators.  Welfare is evaluated on realised tau, V.
    """
    out = {'n_runs': reps, 'n_players': n, 'kappa': DEFAULT_BUDGET,
           'n_mc': ORACLE_MC, 'strategies': {}}
    for nm, runs in runs_by_name.items():
        out['strategies'][nm] = {
            'welfare_mean': float(np.mean([r['welfare'] for r in runs])),
            'welfare_sd': _sd([r['welfare'] for r in runs]),
            'ate_mean_pp': float(np.mean([r['ate'] for r in runs])),
            'ate_sd_pp': _sd([r['ate'] for r in runs]),
            'risk_mean': float(np.mean([r['risk_exposure'] for r in runs])),
            'churn_mod_mean': float(np.mean([r['mean_churn_mod'] for r in runs])),
        }
    return out


def _oracle_budget_table(budget_results, reps, n):
    """Oracle welfare (unrounded) at every budget in BUDGET_FRACTIONS."""
    out = {'n_runs': reps, 'n_players': n, 'n_mc': ORACLE_MC, 'strategies': {}}
    for nm in ORACLE_STRATEGIES:
        out['strategies'][nm] = {}
        for kappa_str, strat_runs in budget_results.items():
            w = [r['welfare'] for r in strat_runs[nm]]
            out['strategies'][nm][kappa_str] = {
                'welfare_mean': float(np.mean(w)), 'welfare_sd': _sd(w)}
    return out


def _aggregate_results(all_results, strategy_names):
    """Aggregate per-run results into summary statistics (unrounded)."""
    agg = {}
    for name in strategy_names:
        runs = all_results[name]
        agg[name] = {
            'ate_mean': float(np.mean([r['ate'] for r in runs])),
            'ate_sd': _sd([r['ate'] for r in runs]),
            'welfare_mean': float(np.mean([r['welfare'] for r in runs])),
            'welfare_sd': _sd([r['welfare'] for r in runs]),
            'risk_mean': float(np.mean([r['risk_exposure'] for r in runs])),
            'risk_sd': _sd([r['risk_exposure'] for r in runs]),
            'mean_churn_mod': float(np.mean([r['mean_churn_mod'] for r in runs])),
            'mean_ltv_cr': float(np.mean([r['mean_ltv_cr'] for r in runs])),
        }
    return agg


EFFICIENCY_RULES = ['CATE Only', 'CATE x LTV', 'Full Framework (Est.)',
                    'Full Framework (Cal.)', 'Full Framework (Cal., plug-in)', 'S-Learner']
EFFICIENCY_ORACLE = 'Oracle feasible (g=1)'


def _efficiency_table(all_results, reps, n):
    """Per-replication paired efficiency against the feasible oracle (g=1) at the default
    budget.  The replications are shared across strategies, so each ratio is formed within a
    replication and then averaged (mean of ratios, t-interval), never a ratio of means.

    level_ratio = W_rule / W_oracle;  normalized = (W_rule - W_random) / (W_oracle - W_random)
    (0 = random targeting, 1 = the feasible oracle).  Every replication must have a positive
    denominator (about 6,100 against an SD of about 480 at full size); otherwise this raises.
    'feasible_over_hindsight' = W_feasible / W_hindsight."""
    def w(name):
        return np.array([r['welfare'] for r in all_results[name]], dtype=float)

    oracle, rand, hind = w(EFFICIENCY_ORACLE), w('Random'), w('Oracle hindsight (g=1)')
    out = {'n_runs': reps, 'n_players': n, 'kappa': DEFAULT_BUDGET, 'oracle': EFFICIENCY_ORACLE,
           'definition': 'mean over replications of the per-replication ratio',
           'rules': {}}
    if not (oracle > 0).all() or not (oracle - rand > 0).all() or not (hind > 0).all():
        raise ValueError('efficiency: a replication has a non-positive feasible-oracle welfare, '
                         'feasible minus random, or hindsight welfare; the ratios would be '
                         'undefined and no replication may be dropped silently')
    for rule in EFFICIENCY_RULES:
        wr = w(rule)
        out['rules'][rule] = {
            'level_ratio': t_mean_ci(wr / oracle),
            'normalized': t_mean_ci((wr - rand) / (oracle - rand)),
        }
    out['feasible_over_hindsight'] = t_mean_ci(oracle / hind)
    return out


def _budget_paired(strat_runs, kappa_str):
    """Cal. vs CATE x LTV and vs CATE Only at one budget; unrounded, with Holm run on the
    unrounded raw p-values (a p of 0.00096 must stay below 0.001)."""
    w_cal = [r['welfare'] for r in strat_runs['Full Framework (Cal.)']]
    w_ltv = [r['welfare'] for r in strat_runs['CATE x LTV']]
    w_cate = [r['welfare'] for r in strat_runs['CATE Only']]
    bp = {
        'cal_vs_ltv': paired_test_raw(w_cal, w_ltv, f'{kappa_str} cal vs ltv'),
        'cal_vs_cate_only': paired_test_raw(w_cal, w_cate, f'{kappa_str} cal vs cate'),
    }
    corrected = holm_bonferroni([bp['cal_vs_ltv']['p_value'], bp['cal_vs_cate_only']['p_value']])
    bp['cal_vs_ltv']['p_holm'] = float(corrected[0])
    bp['cal_vs_cate_only']['p_holm'] = float(corrected[1])
    return bp


def _run_batch(reps, n_jobs=1, **kwargs):
    """run_single_simulation for replications 0..reps-1 (seed SEED_BASE + 7r), in replication
    order whatever n_jobs is."""
    return Parallel(n_jobs=n_jobs)(
        delayed(run_single_simulation)(SEED_BASE + run * 7, **kwargs) for run in range(reps))


MAIN_STRATEGIES = ['Random', 'CATE Only', 'CATE x LTV',
                   'Full Framework (Est.)', 'Full Framework (Cal.)',
                   'Full Framework (Cal., plug-in)',
                   'Oracle hindsight (theta0)', 'S-Learner'] + ORACLE_STRATEGIES


def run_main(reps=None, n=None, kappa=DEFAULT_BUDGET):
    """Main DGP + T-Learner at one budget; returns the aggregated main_results.

    reps and n default to the module-level N_RUNS / N_PLAYERS.
    """
    reps = N_RUNS if reps is None else reps
    all_results = {s: [] for s in MAIN_STRATEGIES}
    for run in range(reps):
        seed = SEED_BASE + run * 7
        results = run_single_simulation(
            seed, budget_frac=kappa, n_players=n)[0]
        for name, res in results.items():
            all_results[name].append(res)
    return _aggregate_results(all_results, MAIN_STRATEGIES)


def run_study(reps=None, n=None, n_jobs=1):
    """Run every scenario and return the output dict (written by main as main.json)."""
    reps = N_RUNS if reps is None else reps
    n = N_PLAYERS if n is None else n
    print("=" * 70)
    print("SIMULATION STUDY: Risk-Modulated Targeting")
    print("=" * 70)
    print(f"N={n}, Runs={reps}, Default budget={DEFAULT_BUDGET*100:.0f}%")
    print(f"Calibration grid: {len(THETA_GRID)} configs (including g~1 identity)")

    strategy_names = MAIN_STRATEGIES

    output = {}

    # =====================================================================
    # 1. Main DGP + T-Learner
    # =====================================================================
    print("\n--- Scenario 1: Main DGP, T-Learner ---")
    all_results = {s: [] for s in strategy_names}
    all_frontiers = []
    all_misspec = []
    all_cal_configs = []

    for results, frontier, misspec_inv, cal_cfg in _run_batch(reps, n_jobs, n_players=n):
        for name, res in results.items():
            all_results[name].append(res)
        all_frontiers.append(frontier)
        all_misspec.append(misspec_inv)
        all_cal_configs.append(cal_cfg)

    output['main_results'] = _aggregate_results(all_results, strategy_names)
    output['efficiency'] = _efficiency_table(all_results, reps, n)

    # Frontier
    n_configs = len(all_frontiers[0])
    output['frontier'] = []
    for i in range(n_configs):
        cfg_name = all_frontiers[0][i]['config']
        output['frontier'].append({
            'config': cfg_name,
            'welfare_mean': round(np.mean([f[i]['welfare'] for f in all_frontiers]), 0),
            'welfare_sd': _sd([f[i]['welfare'] for f in all_frontiers], 0),
            'risk_mean': round(np.mean([f[i]['risk_exposure'] for f in all_frontiers]), 3),
            'risk_sd': _sd([f[i]['risk_exposure'] for f in all_frontiers], 3),
            'mean_churn_mod': round(np.mean([f[i]['mean_churn_mod'] for f in all_frontiers]), 3),
        })

    output['misspecification'] = {
        'inverted_g': {
            'welfare_mean': round(np.mean([m['welfare'] for m in all_misspec]), 0),
            'risk_mean': round(np.mean([m['risk_exposure'] for m in all_misspec]), 3),
        }
    }

    # Calibration summary
    config_strs = [str(sorted(c.items())) for c in all_cal_configs]
    output['calibration_summary'] = {
        'most_common_configs': {k: v for k, v in Counter(config_strs).most_common(3)},
    }

    # Paired tests with Holm-Bonferroni [M3]
    comparisons = [
        ('CATE Only', 'Random'),
        ('CATE x LTV', 'Random'),
        ('Full Framework (Cal.)', 'CATE x LTV'),
        ('Full Framework (Cal.)', 'CATE Only'),
        ('Full Framework (Est.)', 'CATE x LTV'),
        ('Oracle hindsight (theta0)', 'CATE x LTV'),
    ]
    paired_tests = {}
    raw_pvalues = []
    for s1, s2 in comparisons:
        w1 = [r['welfare'] for r in all_results[s1]]
        w2 = [r['welfare'] for r in all_results[s2]]
        label = f'{s1} vs {s2}'
        paired_tests[label] = paired_test_raw(w1, w2, label)
        # mean(a - b) / mean(b), delta-method CI from the per-replication pairs
        paired_tests[label]['ratio_to_b'] = ratio_entry(np.array(w1) - np.array(w2), w2)
        raw_pvalues.append(paired_tests[label]['p_value'])

    corrected = holm_bonferroni(raw_pvalues)
    for i, (s1, s2) in enumerate(comparisons):
        label = f'{s1} vs {s2}'
        paired_tests[label]['p_holm'] = float(corrected[i])
    output['paired_tests'] = paired_tests

    # =====================================================================
    # 2. Main DGP + X-Learner (robustness)
    # =====================================================================
    print("\n--- Scenario 2: Main DGP, X-Learner ---")
    xl_results = {s: [] for s in strategy_names}
    for results, _, _, _ in _run_batch(reps, n_jobs, n_players=n, cate_method='xlearner'):
        for name, res in results.items():
            xl_results[name].append(res)
    output['xlearner_results'] = _aggregate_results(xl_results, strategy_names)

    # Key paired test for X-Learner
    w_cal_xl = [r['welfare'] for r in xl_results['Full Framework (Cal.)']]
    w_cate_xl = [r['welfare'] for r in xl_results['CATE Only']]
    w_ltv_xl = [r['welfare'] for r in xl_results['CATE x LTV']]
    output['xlearner_paired'] = {
        'cal_vs_cate_only': paired_test_raw(w_cal_xl, w_cate_xl, 'XL cal vs cate'),
        'cal_vs_ltv': paired_test_raw(w_cal_xl, w_ltv_xl, 'XL cal vs ltv'),
    }

    # =====================================================================
    # 3. CATE Misspecification + T-Learner
    # =====================================================================
    print("\n--- Scenario 3: CATE Misspecification, T-Learner ---")
    misspec_strats = ['CATE Only', 'CATE x LTV',
                      'Full Framework (Est.)', 'Full Framework (Cal.)']
    misspec_results = {s: [] for s in misspec_strats}
    for results, _, _, _ in _run_batch(reps, n_jobs, n_players=n,
                                       est_func=estimate_components_misspec):
        for name in misspec_strats:
            misspec_results[name].append(results[name])

    output['cate_misspec'] = {}
    for name in misspec_strats:
        runs = misspec_results[name]
        output['cate_misspec'][name] = {
            'welfare_mean': round(np.mean([r['welfare'] for r in runs]), 0),
            'welfare_sd': _sd([r['welfare'] for r in runs], 0),
            'ate_mean': round(np.mean([r['ate'] for r in runs]), 1),
        }

    # Paired tests under misspecification
    w_cal_ms = [r['welfare'] for r in misspec_results['Full Framework (Cal.)']]
    w_cate_ms = [r['welfare'] for r in misspec_results['CATE Only']]
    w_ltv_ms = [r['welfare'] for r in misspec_results['CATE x LTV']]

    ms_paired = {
        'cal_vs_cate': paired_test_raw(w_cal_ms, w_cate_ms, 'ms cal vs cate'),
        'cal_vs_ltv': paired_test_raw(w_cal_ms, w_ltv_ms, 'ms cal vs ltv'),
    }
    ms_raw_p = [ms_paired['cal_vs_cate']['p_value'],
                ms_paired['cal_vs_ltv']['p_value']]
    ms_corrected = holm_bonferroni(ms_raw_p)
    ms_paired['cal_vs_cate']['p_holm'] = float(ms_corrected[0])
    ms_paired['cal_vs_ltv']['p_holm'] = float(ms_corrected[1])

    # TOST for equivalence claim [M3]
    cate_only_mean = np.mean([r['welfare'] for r in misspec_results['CATE Only']])
    tost_margin = 0.05 * abs(cate_only_mean)  # 5% of CATE Only welfare
    ms_paired['tost_cal_vs_cate'] = tost_paired(w_cal_ms, w_cate_ms, tost_margin)
    output['cate_misspec_paired'] = ms_paired

    # =====================================================================
    # 4. CATE Misspecification + X-Learner (robustness)
    # =====================================================================
    print("\n--- Scenario 4: CATE Misspecification, X-Learner ---")
    xl_misspec_results = {s: [] for s in misspec_strats}
    for results, _, _, _ in _run_batch(reps, n_jobs, n_players=n,
                                       est_func=estimate_components_misspec,
                                       cate_method='xlearner'):
        for name in misspec_strats:
            xl_misspec_results[name].append(results[name])

    output['xlearner_misspec'] = {}
    for name in misspec_strats:
        runs = xl_misspec_results[name]
        output['xlearner_misspec'][name] = {
            'welfare_mean': round(np.mean([r['welfare'] for r in runs]), 0),
            'welfare_sd': _sd([r['welfare'] for r in runs], 0),
            'ate_mean': round(np.mean([r['ate'] for r in runs]), 1),
        }
    w_cal_xlm = [r['welfare'] for r in xl_misspec_results['Full Framework (Cal.)']]
    w_cate_xlm = [r['welfare'] for r in xl_misspec_results['CATE Only']]
    output['xlearner_misspec_paired'] = {
        'cal_vs_cate': paired_test_raw(w_cal_xlm, w_cate_xlm, 'xlm cal vs cate'),
    }

    # =====================================================================
    # 5. Null DGP
    # =====================================================================
    print("\n--- Scenario 5: Null DGP (no CATE-churn interaction) ---")
    null_strats = ['CATE Only', 'CATE x LTV',
                   'Full Framework (Est.)', 'Full Framework (Cal.)']
    null_results = {s: [] for s in null_strats}
    for results, _, _, _ in _run_batch(reps, n_jobs, n_players=n,
                                       dgp_func=generate_population_null):
        for name in null_strats:
            null_results[name].append(results[name])

    output['null_dgp'] = {}
    for name in null_strats:
        runs = null_results[name]
        output['null_dgp'][name] = {
            'welfare_mean': round(np.mean([r['welfare'] for r in runs]), 0),
            'welfare_sd': _sd([r['welfare'] for r in runs], 0),
        }
    w_cal_null = [r['welfare'] for r in null_results['Full Framework (Cal.)']]
    w_ltv_null = [r['welfare'] for r in null_results['CATE x LTV']]
    t_null, p_null = scipy_stats.ttest_rel(w_cal_null, w_ltv_null)
    output['null_dgp']['paired_test_cal_vs_ltv'] = {
        'mean_diff': round(float(np.mean(np.array(w_cal_null) - np.array(w_ltv_null))), 0),
        't_stat': round(float(t_null), 2),
        'p_value': round(float(p_null), 4),
    }

    # =====================================================================
    # 6. Correlated-feature DGP [M9]
    # =====================================================================
    print("\n--- Scenario 6: Correlated-feature DGP ---")
    corr_strats = ['CATE Only', 'CATE x LTV',
                   'Full Framework (Est.)', 'Full Framework (Cal.)']
    corr_results = {s: [] for s in corr_strats}
    for results, _, _, _ in _run_batch(reps, n_jobs, n_players=n,
                                       dgp_func=generate_population_correlated):
        for name in corr_strats:
            corr_results[name].append(results[name])

    output['correlated_dgp'] = {}
    for name in corr_strats:
        runs = corr_results[name]
        output['correlated_dgp'][name] = {
            'welfare_mean': round(np.mean([r['welfare'] for r in runs]), 0),
            'welfare_sd': _sd([r['welfare'] for r in runs], 0),
            'ate_mean': round(np.mean([r['ate'] for r in runs]), 1),
        }
    w_cal_corr = [r['welfare'] for r in corr_results['Full Framework (Cal.)']]
    w_cate_corr = [r['welfare'] for r in corr_results['CATE Only']]
    w_ltv_corr = [r['welfare'] for r in corr_results['CATE x LTV']]
    output['correlated_dgp_paired'] = {
        'cal_vs_cate': paired_test_raw(w_cal_corr, w_cate_corr, 'corr cal vs cate'),
        'cal_vs_ltv': paired_test_raw(w_cal_corr, w_ltv_corr, 'corr cal vs ltv'),
    }

    # =====================================================================
    # 7. Budget sensitivity (T-Learner only)
    # =====================================================================
    print("\n--- Budget sensitivity ---")
    budget_results = {}
    budget_paired = {}
    for kappa in BUDGET_FRACTIONS:
        print(f"  kappa={kappa}...")
        budget_results[str(kappa)] = {s: [] for s in strategy_names}
        for results, _, _, _ in _run_batch(reps, n_jobs, n_players=n, budget_frac=kappa):
            for name, res in results.items():
                budget_results[str(kappa)][name].append(res)

    output['budget_sensitivity'] = {}
    for kappa_str, strat_runs in budget_results.items():
        output['budget_sensitivity'][kappa_str] = {}
        for name in strategy_names:
            if strat_runs[name]:
                output['budget_sensitivity'][kappa_str][name] = {
                    'welfare_mean': float(np.mean([r['welfare'] for r in strat_runs[name]])),
                    'ate_mean': float(np.mean([r['ate'] for r in strat_runs[name]])),
                }

    output['oracle_main'] = _oracle_table(
        {nm: all_results[nm] for nm in ORACLE_STRATEGIES}, reps, n)
    output['oracle_budget'] = _oracle_budget_table(budget_results, reps, n)

    # Budget paired tests with Holm-Bonferroni
    for kappa_str, strat_runs in budget_results.items():
        budget_paired[kappa_str] = _budget_paired(strat_runs, kappa_str)
    output['budget_paired_tests'] = budget_paired

    # =====================================================================
    # Metadata
    # =====================================================================
    output['dgp'] = DGP_SPEC
    output['design'] = {
        'n_players': n, 'n_runs': reps,
        'default_budget': DEFAULT_BUDGET, 'treatment_prob': TREATMENT_PROB,
        'budget_sensitivity_runs': reps,
        'calibration_grid_size': len(THETA_GRID),
        'cate_estimators': ['T-Learner', 'X-Learner'],
        'corrections': ['Holm-Bonferroni', 'TOST (equivalence)'],
    }

    # =====================================================================
    # Print summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("MAIN RESULTS (30% budget, T-Learner)")
    print("=" * 70)
    fmt = "{:<26} {:>8} {:>12} {:>8} {:>8}"
    print(fmt.format("Strategy", "ATE(pp)", "Welfare", "Risk", "ChMod"))
    print("-" * 62)
    for name in strategy_names:
        r = output['main_results'][name]
        print(fmt.format(name, f"{r['ate_mean']:.1f}({r['ate_sd']:.1f})",
                         f"{r['welfare_mean']:,.0f}", f"{r['risk_mean']:.3f}",
                         f"{r['mean_churn_mod']:.3f}"))

    print("\n--- Key paired tests (Holm-corrected) ---")
    for label, pt in output['paired_tests'].items():
        print(f"  {label}: diff={pt['mean_diff']:+,.0f}, t={pt['t_stat']:.2f}, "
              f"p={pt['p_value']:.4f}, p_holm={pt['p_holm']:.4f}, "
              f"d={pt['cohens_d']:.3f}, CI={pt['ci_95']}")

    print("\n--- X-Learner results ---")
    for name in strategy_names:
        r = output['xlearner_results'][name]
        print(f"  {name:<26} W={r['welfare_mean']:>10,.0f}")

    print("\n--- CATE Misspecification (T-Learner) ---")
    for name in misspec_strats:
        ms = output['cate_misspec'][name]
        print(f"  {name:<26} W={ms['welfare_mean']:>10,.0f}  ATE={ms['ate_mean']:.1f}")
    mc = output['cate_misspec_paired']
    print(f"  Cal vs CATE Only: diff={mc['cal_vs_cate']['mean_diff']:+,.0f}, "
          f"p={mc['cal_vs_cate']['p_value']:.4f}, "
          f"p_holm={mc['cal_vs_cate']['p_holm']:.4f}")
    tost = mc.get('tost_cal_vs_cate', {})
    print(f"  TOST equivalence (margin={tost.get('margin', 0):,.0f}): "
          f"p_tost={tost.get('p_tost', 1):.4f}, "
          f"equivalent={tost.get('equivalent', False)}")

    print("\n--- Null DGP ---")
    for name in null_strats:
        if isinstance(output['null_dgp'][name], dict) and 'welfare_mean' in output['null_dgp'][name]:
            print(f"  {name:<26} W={output['null_dgp'][name]['welfare_mean']:>10,.0f}")

    print("\n--- Correlated DGP ---")
    for name in corr_strats:
        cr = output['correlated_dgp'][name]
        print(f"  {name:<26} W={cr['welfare_mean']:>10,.0f}")

    print("\n--- Budget sensitivity ---")
    for kappa_str in sorted(output['budget_sensitivity'].keys()):
        bs = output['budget_sensitivity'][kappa_str]
        bp = output['budget_paired_tests'].get(kappa_str, {})
        print(f"\n  kappa={kappa_str}:")
        for name in ['CATE Only', 'CATE x LTV',
                     'Full Framework (Est.)', 'Full Framework (Cal.)']:
            if name in bs:
                print(f"    {name:<26} W={bs[name]['welfare_mean']:>10,.0f}")
        cl = bp.get('cal_vs_ltv', {})
        print(f"    Cal vs LTV: diff={cl.get('mean_diff', 0):+,.0f}, "
              f"p_holm={cl.get('p_holm', 1):.4f}")

    return output


def verify_population_moments():
    """Verify hardcoded population moments against large-sample draw."""
    rng = np.random.RandomState(0)
    n = 2_000_000
    age = rng.normal(24, 8, n).clip(1, 60)
    spend = np.exp(rng.normal(3.5, 0.8, n)).clip(5, 500)
    login = rng.poisson(4, n).clip(1, 15).astype(float)
    bonus = rng.normal(3, 1.5, n).clip(0, 10)
    print("Population moment verification (N=2M):")
    for name, vals in [('account_age', age), ('weekly_spend', spend),
                       ('login_freq', login), ('bonus_history', bonus)]:
        print(f"  {name}: mean={vals.mean():.4f} (coded {POP_MEAN[name]}), "
              f"sd={vals.std():.4f} (coded {POP_SD[name]})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_common_args(parser)
    parser.add_argument('--jobs', type=int, default=1, help='parallel worker processes')
    parser.add_argument('--verify-moments', action='store_true',
                        help='also check the hardcoded population moments (N=2M draw)')
    args = parser.parse_args(argv)
    reps, n = resolve_run_size(args, N_RUNS, N_PLAYERS)
    if args.verify_moments:
        verify_population_moments()
    output = run_study(reps, n, n_jobs=args.jobs)
    name = 'main_smoke.json' if args.smoke else 'main.json'
    print(f"\nResults saved to {save_json(output, name)}")


if __name__ == '__main__':
    main()
