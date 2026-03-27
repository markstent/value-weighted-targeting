"""
Simulation study for v27 manuscript.

Fixes over v26:
- Add null DGP (no CATE-churn interaction, m=1) for boundary conditions
- Fix CV to use actual train/validation fold split
- Paired t-tests at ALL budget levels
- Include CATE Only in budget sensitivity table
- LTV proxy noise reduced to SD=150 (was 300, which was too aggressive)
- Rename "CV" to "calibrated" throughout to avoid misleading terminology
"""

import numpy as np
import json
import os
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# --- Configuration ---
N_PLAYERS = 10000
N_RUNS = 200
BUDGET_FRACTIONS = [0.10, 0.20, 0.30, 0.50]
DEFAULT_BUDGET = 0.30
TREATMENT_PROB = 0.50
SEED_BASE = 42


# --- DGP Specification (fully documented) ---
# Features:
#   X1 = account_age ~ N(24, 8), clipped [1, 60]
#   X2 = weekly_spend ~ exp(N(3.5, 0.8)), clipped [5, 500]
#   X3 = login_freq ~ Poisson(4), clipped [1, 15]
#   X4 = bonus_history ~ N(3, 1.5), clipped [0, 10]
#
# Churn probability:
#   logit(p_churn) = -1.0 + 0.04*(30 - X1) - 0.25*X3 + 0.01*(40 - X2) + 0.1*z4 + N(0, 0.4)
#   p_churn = clip(sigmoid(logit), 0.01, 0.99)
#
# LTV:
#   LTV = 10*X2 + 40*X3 + 5*X1 + N(0, 80), clipped [50, 8000]
#
# CATE (with churn interaction):
#   tau_base = 0.06 + 0.10*z3 - 0.02*z3^2 - 0.04*z2
#   churn_moderation = 4 * p_churn * (1 - p_churn)  # inverse-U, peak at p=0.5
#   true_tau = (tau_base * churn_moderation + N(0, 0.03)), clipped [-0.20, 0.40]
#
# Baseline retention:
#   y0_prob = clip(0.6 + 0.15*z3 + 0.05*z2 - 0.08*z1 + N(0, 0.05), 0.05, 0.95)
#
# Potential outcomes:
#   y1_prob = clip(y0_prob + true_tau, 0.01, 0.99)
#   Y(0) ~ Bernoulli(y0_prob), Y(1) ~ Bernoulli(y1_prob)
#   W ~ Bernoulli(0.5), Y_obs = W*Y(1) + (1-W)*Y(0)


DGP_SPEC = {
    'features': {
        'account_age': 'N(24, 8), clipped [1, 60]',
        'weekly_spend': 'exp(N(3.5, 0.8)), clipped [5, 500]',
        'login_freq': 'Poisson(4), clipped [1, 15]',
        'bonus_history': 'N(3, 1.5), clipped [0, 10]',
    },
    'churn': 'logit = -1.0 + 0.04*(30-age) - 0.25*login + 0.01*(40-spend) + 0.1*z_bonus + N(0,0.4); sigmoid, clip [0.01, 0.99]',
    'ltv': '10*spend + 40*login + 5*age + N(0,80), clip [50, 8000]',
    'cate': 'tau_base = 0.06 + 0.10*z_login - 0.02*z_login^2 - 0.04*z_spend; churn_mod = 4*p*(1-p); tau = tau_base * churn_mod + N(0,0.03), clip [-0.20, 0.40]',
    'baseline': '0.6 + 0.15*z_login + 0.05*z_spend - 0.08*z_age + N(0,0.05), clip [0.05, 0.95]',
    'treatment': 'Bernoulli(0.5)',
    'note': 'The CATE-churn moderation (4*p*(1-p)) creates an inverse-U relationship: treatment effects peak at moderate churn (p=0.5) and are attenuated at extremes. This is the structural feature that risk modulation is designed to exploit.',
}


def f_churn(p, pL=0.05, pML=0.40, pMH=0.60, pH=0.90, phi=0.20):
    """Piecewise linear risk-modulation function."""
    if p < pL:
        return 0.0
    elif p < pML:
        return (p - pL) / (pML - pL)
    elif p <= pMH:
        return 1.0
    elif p < pH:
        return 1.0 - (1.0 - phi) * (p - pMH) / (pH - pMH)
    else:
        return phi


def f_churn_vec(p_arr, **kwargs):
    return np.array([f_churn(p, **kwargs) for p in p_arr])


def generate_population(n, rng):
    """Generate player population with CATE-churn inverse-U interaction."""
    # Features
    account_age = rng.normal(24, 8, n).clip(1, 60)
    weekly_spend = np.exp(rng.normal(3.5, 0.8, n)).clip(5, 500)
    login_freq = rng.poisson(4, n).clip(1, 15).astype(float)
    bonus_history = rng.normal(3, 1.5, n).clip(0, 10)

    # Standardised
    z_spend = (weekly_spend - weekly_spend.mean()) / weekly_spend.std()
    z_login = (login_freq - login_freq.mean()) / login_freq.std()
    z_age = (account_age - account_age.mean()) / account_age.std()
    z_bonus = (bonus_history - bonus_history.mean()) / bonus_history.std()

    # Churn probability
    churn_logit = (-1.0 + 0.04 * (30 - account_age) - 0.25 * login_freq
                   + 0.01 * (40 - weekly_spend) + 0.1 * z_bonus
                   + rng.normal(0, 0.4, n))
    true_churn = (1.0 / (1.0 + np.exp(-churn_logit))).clip(0.01, 0.99)

    # LTV
    true_ltv = (weekly_spend * 10 + login_freq * 40 + account_age * 5
                + rng.normal(0, 80, n)).clip(50, 8000)

    # CATE with churn moderation (inverse-U)
    tau_base = 0.06 + 0.10 * z_login - 0.02 * z_login**2 - 0.04 * z_spend
    churn_moderation = 4.0 * true_churn * (1.0 - true_churn)  # peaks at p=0.5
    true_tau = (tau_base * churn_moderation + rng.normal(0, 0.03, n)).clip(-0.20, 0.40)

    # Baseline and potential outcomes
    baseline = (0.6 + 0.15 * z_login + 0.05 * z_spend - 0.08 * z_age
                + rng.normal(0, 0.05, n)).clip(0.05, 0.95)
    y0_prob = baseline
    y1_prob = (baseline + true_tau).clip(0.01, 0.99)
    treatment = rng.binomial(1, TREATMENT_PROB, n)
    y0 = rng.binomial(1, y0_prob)
    y1 = rng.binomial(1, y1_prob)
    y_obs = treatment * y1 + (1 - treatment) * y0

    features = np.column_stack([account_age, weekly_spend, login_freq, bonus_history])
    return {
        'features': features,
        'feature_names': ['account_age', 'weekly_spend', 'login_freq', 'bonus_history'],
        'treatment': treatment, 'y_obs': y_obs,
        'y0': y0, 'y1': y1,
        'true_tau': true_tau, 'true_ltv': true_ltv, 'true_churn': true_churn,
        'y0_prob': y0_prob, 'y1_prob': y1_prob,
    }


def generate_population_null(n, rng):
    """Generate population with NO CATE-churn interaction (m=1).
    CATE depends on features but is independent of churn probability.
    This tests the boundary condition: g should add nothing."""
    account_age = rng.normal(24, 8, n).clip(1, 60)
    weekly_spend = np.exp(rng.normal(3.5, 0.8, n)).clip(5, 500)
    login_freq = rng.poisson(4, n).clip(1, 15).astype(float)
    bonus_history = rng.normal(3, 1.5, n).clip(0, 10)

    z_spend = (weekly_spend - weekly_spend.mean()) / weekly_spend.std()
    z_login = (login_freq - login_freq.mean()) / login_freq.std()
    z_age = (account_age - account_age.mean()) / account_age.std()
    z_bonus = (bonus_history - bonus_history.mean()) / bonus_history.std()

    churn_logit = (-1.0 + 0.04 * (30 - account_age) - 0.25 * login_freq
                   + 0.01 * (40 - weekly_spend) + 0.1 * z_bonus
                   + rng.normal(0, 0.4, n))
    true_churn = (1.0 / (1.0 + np.exp(-churn_logit))).clip(0.01, 0.99)
    true_ltv = (weekly_spend * 10 + login_freq * 40 + account_age * 5
                + rng.normal(0, 80, n)).clip(50, 8000)

    # CATE WITHOUT churn moderation (m=1)
    tau_base = 0.06 + 0.10 * z_login - 0.02 * z_login**2 - 0.04 * z_spend
    true_tau = (tau_base + rng.normal(0, 0.03, n)).clip(-0.20, 0.40)

    baseline = (0.6 + 0.15 * z_login + 0.05 * z_spend - 0.08 * z_age
                + rng.normal(0, 0.05, n)).clip(0.05, 0.95)
    y0_prob = baseline
    y1_prob = (baseline + true_tau).clip(0.01, 0.99)
    treatment = rng.binomial(1, TREATMENT_PROB, n)
    y0 = rng.binomial(1, y0_prob)
    y1 = rng.binomial(1, y1_prob)
    y_obs = treatment * y1 + (1 - treatment) * y0

    features = np.column_stack([account_age, weekly_spend, login_freq, bonus_history])
    return {
        'features': features,
        'feature_names': ['account_age', 'weekly_spend', 'login_freq', 'bonus_history'],
        'treatment': treatment, 'y_obs': y_obs,
        'y0': y0, 'y1': y1,
        'true_tau': true_tau, 'true_ltv': true_ltv, 'true_churn': true_churn,
        'y0_prob': y0_prob, 'y1_prob': y1_prob,
    }


def estimate_components(data, train_idx, test_idx):
    """Estimate CATE, LTV, and churn from training data."""
    X = data['features']
    W = data['treatment']
    Y = data['y_obs']
    X_train, X_test = X[train_idx], X[test_idx]
    W_train, Y_train = W[train_idx], Y[train_idx]

    treat_mask = W_train == 1
    ctrl_mask = W_train == 0

    model_t = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0)
    model_c = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0)
    model_t.fit(X_train[treat_mask], Y_train[treat_mask])
    model_c.fit(X_train[ctrl_mask], Y_train[ctrl_mask])
    tau_hat = (model_t.predict_proba(X_test)[:, 1]
               - model_c.predict_proba(X_test)[:, 1])

    # LTV (proxy with moderate noise; SD=150 adds realistic estimation error
    # without overwhelming the signal, which has SD~80 in the DGP)
    rng_ltv = np.random.RandomState(hash(tuple(train_idx[:5])) % (2**31))
    ltv_proxy = (X_train[:, 1] * 10 + X_train[:, 2] * 40 + X_train[:, 0] * 5
                 + rng_ltv.normal(0, 150, len(X_train)))
    ltv_model = GradientBoostingRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.1, random_state=0)
    ltv_model.fit(X_train, ltv_proxy)
    ltv_hat = ltv_model.predict(X_test).clip(50, 8000)

    # Churn
    churn_proxy = 1 - Y_train[ctrl_mask]
    churn_model = GradientBoostingClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.1, random_state=0)
    churn_model.fit(X_train[ctrl_mask], churn_proxy)
    churn_hat = churn_model.predict_proba(X_test)[:, 1].clip(0.01, 0.99)

    return tau_hat, ltv_hat, churn_hat


def compute_welfare(targeted_idx, true_tau, true_ltv):
    return float(np.sum(true_tau[targeted_idx] * true_ltv[targeted_idx]))


def compute_risk_exposure(targeted_idx, true_churn):
    return float(np.mean(true_churn[targeted_idx]))


def compute_mean_churn_moderation(targeted_idx, true_churn):
    """Mean CATE-churn moderation factor among targeted players."""
    return float(np.mean(4.0 * true_churn[targeted_idx] * (1.0 - true_churn[targeted_idx])))


def target_topk(scores, k):
    return np.argsort(scores)[-k:]


THETA_GRID = [
    {'pL': 0.03, 'pML': 0.30, 'pMH': 0.65, 'phi': 0.15},
    {'pL': 0.05, 'pML': 0.35, 'pMH': 0.60, 'phi': 0.20},
    {'pL': 0.05, 'pML': 0.40, 'pMH': 0.60, 'phi': 0.20},  # default
    {'pL': 0.08, 'pML': 0.30, 'pMH': 0.55, 'phi': 0.10},
    {'pL': 0.10, 'pML': 0.35, 'pMH': 0.55, 'phi': 0.15},
    {'pL': 0.05, 'pML': 0.25, 'pMH': 0.50, 'phi': 0.10},
    {'pL': 0.03, 'pML': 0.20, 'pMH': 0.55, 'phi': 0.10},
    {'pL': 0.05, 'pML': 0.30, 'pMH': 0.55, 'phi': 0.15},
    {'pL': 0.08, 'pML': 0.25, 'pMH': 0.50, 'phi': 0.05},
]


def calibrate_theta(data, train_idx, seed, k):
    """Calibrate theta by held-out validation with grid search.

    Splits training data into estimation (60%) and validation (40%) folds.
    Components are estimated on the estimation fold. Each candidate theta
    is scored on the validation fold using estimated welfare only.
    No oracle information is used at any stage.
    """
    # Split training into estimation and validation
    est_rel, val_rel = train_test_split(
        np.arange(len(train_idx)), test_size=0.40, random_state=seed + 99)
    est_idx = train_idx[est_rel]
    val_idx = train_idx[val_rel]

    # Estimate components on estimation fold, predict on validation fold
    tau_val, ltv_val, churn_val = estimate_components(data, est_idx, val_idx)
    k_val = int(len(val_idx) * DEFAULT_BUDGET)

    best_welfare = -np.inf
    best_config = THETA_GRID[2]  # default fallback
    for cfg in THETA_GRID:
        g_val = f_churn_vec(churn_val, **cfg)
        scores = tau_val * ltv_val * g_val
        targeted = target_topk(scores, k_val)
        # Estimated welfare on validation fold (no oracle)
        w = float(np.sum(tau_val[targeted] * ltv_val[targeted]))
        if w > best_welfare:
            best_welfare = w
            best_config = cfg
    return best_config


def run_single_simulation(seed, budget_frac=DEFAULT_BUDGET, dgp_func=None):
    if dgp_func is None:
        dgp_func = generate_population
    rng = np.random.RandomState(seed)
    data = dgp_func(N_PLAYERS, rng)

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

    tau_hat, ltv_hat, churn_hat = estimate_components(data, train_idx, test_idx)

    g_hat = f_churn_vec(churn_hat)
    g_oracle = f_churn_vec(true_churn)

    # Calibrated theta via held-out validation (no oracle)
    cal_config = calibrate_theta(data, train_idx, seed, k)
    g_cal = f_churn_vec(churn_hat, **cal_config)

    strategies = {}
    strategies['Random'] = rng.uniform(0, 1, len(test_idx))
    strategies['CATE Only'] = tau_hat
    strategies['CATE x LTV'] = tau_hat * ltv_hat
    strategies['Full Framework (Est.)'] = tau_hat * ltv_hat * g_hat
    strategies['Full Framework (Cal.)'] = tau_hat * ltv_hat * g_cal
    strategies['Full Framework (Oracle)'] = data['true_tau'][test_idx] * data['true_ltv'][test_idx] * g_oracle

    # Single end-to-end model baseline: train on estimated composite scores
    # Uses a separate estimation fold to avoid data leakage
    # Split training data into estimation and model-fitting folds
    est_idx, fit_idx = train_test_split(
        np.arange(len(train_idx)), test_size=0.5, random_state=seed + 1)
    est_train = train_idx[est_idx]
    fit_train = train_idx[fit_idx]
    # Estimate components on estimation fold
    tau_est, ltv_est, churn_est = estimate_components(data, est_train, fit_train)
    g_est = f_churn_vec(churn_est)
    est_target = tau_est * ltv_est * g_est
    # Train single model on fitting fold using estimated targets
    single_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0)
    single_model.fit(data['features'][fit_train], est_target)
    strategies['Single Model'] = single_model.predict(data['features'][test_idx])

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
        ('Aggressive (0.15/0.35/0.50/phi=0.05)', {'pL': 0.15, 'pML': 0.35, 'pMH': 0.50, 'phi': 0.05}),
        ('Calibrated', None),  # handled separately
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


def main():
    print("=" * 70)
    print("SIMULATION STUDY v27: Risk-Modulated Targeting with CATE-Churn Interaction")
    print("=" * 70)
    print(f"N={N_PLAYERS}, Runs={N_RUNS}, Default budget={DEFAULT_BUDGET*100:.0f}%")

    strategy_names = ['Random', 'CATE Only', 'CATE x LTV',
                      'Full Framework (Est.)', 'Full Framework (Cal.)',
                      'Full Framework (Oracle)', 'Single Model']
    all_results = {s: [] for s in strategy_names}
    all_frontiers = []
    all_misspec = []
    all_cal_configs = []

    for run in range(N_RUNS):
        if (run + 1) % 20 == 0:
            print(f"  Run {run + 1}/{N_RUNS}...")
        seed = SEED_BASE + run * 7
        results, frontier, misspec, cal_cfg = run_single_simulation(seed)
        for name, res in results.items():
            all_results[name].append(res)
        all_frontiers.append(frontier)
        all_misspec.append(misspec)
        all_cal_configs.append(cal_cfg)

    # --- Budget sensitivity (full N_RUNS for all budget levels) ---
    print("\n  Running budget sensitivity...")
    budget_results = {}
    for kappa in BUDGET_FRACTIONS:
        budget_results[str(kappa)] = {s: [] for s in strategy_names}
        for run in range(N_RUNS):
            seed = SEED_BASE + run * 7
            results, _, _, _ = run_single_simulation(seed, budget_frac=kappa)
            for name, res in results.items():
                budget_results[str(kappa)][name].append(res)

    # --- Aggregate ---
    output = {'main_results': {}}
    for name in strategy_names:
        runs = all_results[name]
        output['main_results'][name] = {
            'ate_mean': round(np.mean([r['ate'] for r in runs]), 1),
            'ate_sd': round(np.std([r['ate'] for r in runs]), 1),
            'welfare_mean': round(np.mean([r['welfare'] for r in runs]), 0),
            'welfare_sd': round(np.std([r['welfare'] for r in runs]), 0),
            'risk_mean': round(np.mean([r['risk_exposure'] for r in runs]), 3),
            'risk_sd': round(np.std([r['risk_exposure'] for r in runs]), 3),
            'mean_churn_mod': round(np.mean([r['mean_churn_mod'] for r in runs]), 3),
            'mean_ltv_cr': round(np.mean([r['mean_ltv_cr'] for r in runs]), 0),
        }

    # Frontier
    n_configs = len(all_frontiers[0])
    output['frontier'] = []
    for i in range(n_configs):
        cfg_name = all_frontiers[0][i]['config']
        output['frontier'].append({
            'config': cfg_name,
            'welfare_mean': round(np.mean([f[i]['welfare'] for f in all_frontiers]), 0),
            'welfare_sd': round(np.std([f[i]['welfare'] for f in all_frontiers]), 0),
            'risk_mean': round(np.mean([f[i]['risk_exposure'] for f in all_frontiers]), 3),
            'risk_sd': round(np.std([f[i]['risk_exposure'] for f in all_frontiers]), 3),
            'mean_churn_mod': round(np.mean([f[i]['mean_churn_mod'] for f in all_frontiers]), 3),
        })

    # Misspecification
    output['misspecification'] = {
        'inverted_g': {
            'welfare_mean': round(np.mean([m['welfare'] for m in all_misspec]), 0),
            'risk_mean': round(np.mean([m['risk_exposure'] for m in all_misspec]), 3),
        }
    }

    # Budget sensitivity
    output['budget_sensitivity'] = {}
    for kappa_str, strat_runs in budget_results.items():
        output['budget_sensitivity'][kappa_str] = {}
        for name in strategy_names:
            if strat_runs[name]:
                output['budget_sensitivity'][kappa_str][name] = {
                    'welfare_mean': round(np.mean([r['welfare'] for r in strat_runs[name]]), 0),
                    'ate_mean': round(np.mean([r['ate'] for r in strat_runs[name]]), 1),
                }

    # CV config summary
    output['calibration_summary'] = {
        'most_common_configs': {},
    }
    config_strs = [str(sorted(c.items())) for c in all_cal_configs]
    from collections import Counter
    for cfg_str, count in Counter(config_strs).most_common(3):
        output['calibration_summary']['most_common_configs'][cfg_str] = count

    # Paired t-tests for key comparisons
    from scipy import stats as scipy_stats
    paired_tests = {}
    comparisons = [
        ('CATE Only', 'Random'),
        ('CATE x LTV', 'Random'),
        ('Full Framework (Cal.)', 'CATE x LTV'),
        ('Full Framework (Cal.)', 'CATE Only'),
        ('Full Framework (Est.)', 'CATE x LTV'),
        ('Full Framework (Oracle)', 'CATE x LTV'),
    ]
    for s1, s2 in comparisons:
        w1 = [r['welfare'] for r in all_results[s1]]
        w2 = [r['welfare'] for r in all_results[s2]]
        diff = np.array(w1) - np.array(w2)
        t_stat, p_val = scipy_stats.ttest_rel(w1, w2)
        paired_tests[f'{s1} vs {s2}'] = {
            'mean_diff': round(float(np.mean(diff)), 0),
            'se_diff': round(float(np.std(diff) / np.sqrt(len(diff))), 0),
            't_stat': round(float(t_stat), 2),
            'p_value': round(float(p_val), 4),
        }
    output['paired_tests'] = paired_tests

    # Paired t-tests at each budget level (Cal. vs CATE x LTV)
    budget_paired = {}
    for kappa_str, strat_runs in budget_results.items():
        w_cal = [r['welfare'] for r in strat_runs['Full Framework (Cal.)']]
        w_ltv = [r['welfare'] for r in strat_runs['CATE x LTV']]
        w_cate = [r['welfare'] for r in strat_runs['CATE Only']]
        diff_ltv = np.array(w_cal) - np.array(w_ltv)
        diff_cate = np.array(w_cal) - np.array(w_cate)
        t_ltv, p_ltv = scipy_stats.ttest_rel(w_cal, w_ltv)
        t_cate, p_cate = scipy_stats.ttest_rel(w_cal, w_cate)
        budget_paired[kappa_str] = {
            'cal_vs_ltv': {
                'mean_diff': round(float(np.mean(diff_ltv)), 0),
                't_stat': round(float(t_ltv), 2),
                'p_value': round(float(p_ltv), 4),
            },
            'cal_vs_cate_only': {
                'mean_diff': round(float(np.mean(diff_cate)), 0),
                't_stat': round(float(t_cate), 2),
                'p_value': round(float(p_cate), 4),
            },
        }
    output['budget_paired_tests'] = budget_paired

    # --- Null DGP (no CATE-churn interaction) ---
    print("\n  Running null DGP (no CATE-churn interaction)...")
    null_results = {s: [] for s in ['CATE Only', 'CATE x LTV',
                                     'Full Framework (Est.)', 'Full Framework (Cal.)']}
    for run in range(N_RUNS):
        if (run + 1) % 20 == 0:
            print(f"    Null DGP run {run + 1}/{N_RUNS}...")
        seed = SEED_BASE + run * 7
        results, _, _, _ = run_single_simulation(seed, dgp_func=generate_population_null)
        for name in null_results:
            null_results[name].append(results[name])
    output['null_dgp'] = {}
    for name in null_results:
        runs = null_results[name]
        output['null_dgp'][name] = {
            'welfare_mean': round(np.mean([r['welfare'] for r in runs]), 0),
            'welfare_sd': round(np.std([r['welfare'] for r in runs]), 0),
        }
    # Paired test: Cal. vs CATE x LTV under null DGP
    w_cal_null = [r['welfare'] for r in null_results['Full Framework (Cal.)']]
    w_ltv_null = [r['welfare'] for r in null_results['CATE x LTV']]
    t_null, p_null = scipy_stats.ttest_rel(w_cal_null, w_ltv_null)
    output['null_dgp']['paired_test_cal_vs_ltv'] = {
        'mean_diff': round(float(np.mean(np.array(w_cal_null) - np.array(w_ltv_null))), 0),
        't_stat': round(float(t_null), 2),
        'p_value': round(float(p_null), 4),
    }

    output['dgp'] = DGP_SPEC
    output['design'] = {
        'n_players': N_PLAYERS, 'n_runs': N_RUNS,
        'default_budget': DEFAULT_BUDGET, 'treatment_prob': TREATMENT_PROB,
        'budget_sensitivity_runs': N_RUNS,
    }

    # Print
    print("\n" + "=" * 70)
    print("MAIN RESULTS (30% budget)")
    print("=" * 70)
    fmt = "{:<26} {:>8} {:>12} {:>8} {:>8}"
    print(fmt.format("Strategy", "ATE(pp)", "Welfare", "Risk", "ChMod"))
    print("-" * 62)
    for name in strategy_names:
        r = output['main_results'][name]
        print(fmt.format(name, f"{r['ate_mean']:.1f}({r['ate_sd']:.1f})",
                         f"{r['welfare_mean']:,.0f}", f"{r['risk_mean']:.3f}",
                         f"{r['mean_churn_mod']:.3f}"))

    print("\n" + "=" * 70)
    print("VALUE-CONSERVATISM FRONTIER")
    print("=" * 70)
    for f in output['frontier']:
        print(f"  {f['config']:<36} W={f['welfare_mean']:>10,.0f}  "
              f"R={f['risk_mean']:.3f}  ChMod={f['mean_churn_mod']:.3f}")

    print(f"\nMisspec (inverted g): W={output['misspecification']['inverted_g']['welfare_mean']:,.0f}")

    print("\n" + "=" * 70)
    print("BUDGET SENSITIVITY (with paired tests)")
    print("=" * 70)
    for kappa_str in sorted(output['budget_sensitivity'].keys()):
        bs = output['budget_sensitivity'][kappa_str]
        bp = output['budget_paired_tests'].get(kappa_str, {})
        print(f"\n  kappa={kappa_str}:")
        for name in ['CATE Only', 'CATE x LTV', 'Full Framework (Est.)', 'Full Framework (Cal.)']:
            if name in bs:
                print(f"    {name:<26} W={bs[name]['welfare_mean']:>10,.0f}  "
                      f"ATE={bs[name]['ate_mean']:.1f}")
        if bp:
            cl = bp.get('cal_vs_ltv', {})
            cc = bp.get('cal_vs_cate_only', {})
            print(f"    Cal vs LTV: diff={cl.get('mean_diff',0):+,.0f}, "
                  f"t={cl.get('t_stat',0):.2f}, p={cl.get('p_value',1):.4f}")
            print(f"    Cal vs CATE: diff={cc.get('mean_diff',0):+,.0f}, "
                  f"t={cc.get('t_stat',0):.2f}, p={cc.get('p_value',1):.4f}")

    print("\n" + "=" * 70)
    print("NULL DGP (no CATE-churn interaction)")
    print("=" * 70)
    for name in output['null_dgp']:
        if isinstance(output['null_dgp'][name], dict) and 'welfare_mean' in output['null_dgp'][name]:
            print(f"  {name:<26} W={output['null_dgp'][name]['welfare_mean']:>10,.0f}")
    nt = output['null_dgp'].get('paired_test_cal_vs_ltv', {})
    print(f"  Cal vs LTV: diff={nt.get('mean_diff',0):+,.0f}, "
          f"t={nt.get('t_stat',0):.2f}, p={nt.get('p_value',1):.4f}")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'simulation_v27.json')
    with open(output_path, 'w') as fp:
        json.dump(output, fp, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
