"""
Simulation study for v29 manuscript.

Changes over v28:
- Fixed: within-sample standardisation replaced with population parameters [M1]
- Fixed: LTV proxy now uses misspecified functional form [C2]
- Added: X-Learner CATE estimator for robustness [M4]
- Added: S-Learner baseline replacing Single Model [C3]
- Added: Correlated-feature DGP scenario [M9]
- Added: Expanded calibration grid (22 configs including g~1) [M15]
- Added: Holm-Bonferroni correction, TOST equivalence tests, Cohen's d [M3]
"""

import numpy as np
import json
import os
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scipy import stats as scipy_stats

# --- Configuration ---
N_PLAYERS = 10000
N_RUNS = 200
BUDGET_FRACTIONS = [0.10, 0.20, 0.30, 0.50]
DEFAULT_BUDGET = 0.30
TREATMENT_PROB = 0.50
SEED_BASE = 42

# --- Population moments (verified via N=2M draw, fixed for reproducibility) ---
# These replace within-sample standardisation to prevent train/test leakage.
POP_MEAN = {
    'account_age': 24.01,
    'weekly_spend': 45.56,
    'login_freq': 4.02,
    'bonus_history': 3.01,
}
POP_SD = {
    'account_age': 7.98,
    'weekly_spend': 42.51,
    'login_freq': 1.97,
    'bonus_history': 1.47,
}


# --- DGP Specification ---
DGP_SPEC = {
    'features': {
        'account_age': 'N(24, 8), clipped [1, 60]',
        'weekly_spend': 'exp(N(3.5, 0.8)), clipped [5, 500]',
        'login_freq': 'Poisson(4), clipped [1, 15]',
        'bonus_history': 'N(3, 1.5), clipped [0, 10]',
    },
    'standardisation': 'Population parameters (fixed across replications)',
    'churn': 'logit = -1.0 + 0.04*(30-age) - 0.25*login + 0.01*(40-spend) + 0.1*z_bonus + N(0,0.4); sigmoid, clip [0.01, 0.99]',
    'ltv': '10*spend + 40*login + 5*age + N(0,80), clip [50, 8000]',
    'ltv_proxy': '12*spend^0.9 + 35*login + N(0,200) -- misspecified form for estimation',
    'cate': 'tau_base = 0.06 + 0.10*z_login - 0.02*z_login^2 - 0.04*z_spend; churn_mod = 4*p*(1-p); tau = tau_base * churn_mod + N(0,0.03), clip [-0.20, 0.40]',
    'baseline': '0.6 + 0.15*z_login + 0.05*z_spend - 0.08*z_age + N(0,0.05), clip [0.05, 0.95]',
    'treatment': 'Bernoulli(0.5)',
    'note': 'The CATE-churn moderation (4*p*(1-p)) creates an inverse-U relationship: treatment effects peak at moderate churn (p=0.5) and are attenuated at extremes.',
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


def _standardise(values, feature_name):
    """Standardise using fixed population parameters."""
    return (values - POP_MEAN[feature_name]) / POP_SD[feature_name]


def generate_population(n, rng):
    """Generate player population with CATE-churn inverse-U interaction."""
    account_age = rng.normal(24, 8, n).clip(1, 60)
    weekly_spend = np.exp(rng.normal(3.5, 0.8, n)).clip(5, 500)
    login_freq = rng.poisson(4, n).clip(1, 15).astype(float)
    bonus_history = rng.normal(3, 1.5, n).clip(0, 10)

    z_spend = _standardise(weekly_spend, 'weekly_spend')
    z_login = _standardise(login_freq, 'login_freq')
    z_age = _standardise(account_age, 'account_age')
    z_bonus = _standardise(bonus_history, 'bonus_history')

    churn_logit = (-1.0 + 0.04 * (30 - account_age) - 0.25 * login_freq
                   + 0.01 * (40 - weekly_spend) + 0.1 * z_bonus
                   + rng.normal(0, 0.4, n))
    true_churn = (1.0 / (1.0 + np.exp(-churn_logit))).clip(0.01, 0.99)

    true_ltv = (weekly_spend * 10 + login_freq * 40 + account_age * 5
                + rng.normal(0, 80, n)).clip(50, 8000)

    tau_base = 0.06 + 0.10 * z_login - 0.02 * z_login**2 - 0.04 * z_spend
    churn_moderation = 4.0 * true_churn * (1.0 - true_churn)
    true_tau = (tau_base * churn_moderation + rng.normal(0, 0.03, n)).clip(-0.20, 0.40)

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
    """Generate population with NO CATE-churn interaction (m=1)."""
    account_age = rng.normal(24, 8, n).clip(1, 60)
    weekly_spend = np.exp(rng.normal(3.5, 0.8, n)).clip(5, 500)
    login_freq = rng.poisson(4, n).clip(1, 15).astype(float)
    bonus_history = rng.normal(3, 1.5, n).clip(0, 10)

    z_spend = _standardise(weekly_spend, 'weekly_spend')
    z_login = _standardise(login_freq, 'login_freq')
    z_age = _standardise(account_age, 'account_age')
    z_bonus = _standardise(bonus_history, 'bonus_history')

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


def generate_population_correlated(n, rng):
    """Generate population with correlated features via Gaussian copula [M9]."""
    # Correlation matrix for underlying normals
    corr = np.array([
        [1.0,  0.3,  0.4,  0.0],   # age
        [0.3,  1.0,  0.0,  0.0],   # spend (underlying normal)
        [0.4,  0.0,  1.0,  0.3],   # login (underlying normal)
        [0.0,  0.0,  0.3,  1.0],   # bonus
    ])
    L = np.linalg.cholesky(corr)
    Z = rng.normal(0, 1, (n, 4))
    Z_corr = Z @ L.T

    account_age = (24 + 8 * Z_corr[:, 0]).clip(1, 60)
    weekly_spend = np.exp(3.5 + 0.8 * Z_corr[:, 1]).clip(5, 500)
    # Map correlated normal to Poisson-like via rounding of exp transform
    login_freq = np.round(np.exp(np.log(4) + 0.45 * Z_corr[:, 2])).clip(1, 15).astype(float)
    bonus_history = (3 + 1.5 * Z_corr[:, 3]).clip(0, 10)

    z_spend = _standardise(weekly_spend, 'weekly_spend')
    z_login = _standardise(login_freq, 'login_freq')
    z_age = _standardise(account_age, 'account_age')
    z_bonus = _standardise(bonus_history, 'bonus_history')

    churn_logit = (-1.0 + 0.04 * (30 - account_age) - 0.25 * login_freq
                   + 0.01 * (40 - weekly_spend) + 0.1 * z_bonus
                   + rng.normal(0, 0.4, n))
    true_churn = (1.0 / (1.0 + np.exp(-churn_logit))).clip(0.01, 0.99)
    true_ltv = (weekly_spend * 10 + login_freq * 40 + account_age * 5
                + rng.normal(0, 80, n)).clip(50, 8000)

    tau_base = 0.06 + 0.10 * z_login - 0.02 * z_login**2 - 0.04 * z_spend
    churn_moderation = 4.0 * true_churn * (1.0 - true_churn)
    true_tau = (tau_base * churn_moderation + rng.normal(0, 0.03, n)).clip(-0.20, 0.40)

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


# --- Estimation functions ---

def _estimate_ltv_churn(X_train, X_test, W_train, Y_train, train_idx):
    """Shared LTV and churn estimation (used by all CATE methods)."""
    # LTV: misspecified proxy (drops age, nonlinear spend, different coefficients)
    rng_ltv = np.random.RandomState(hash(tuple(train_idx[:5])) % (2**31))
    ltv_proxy = (12 * X_train[:, 1]**0.9 + 35 * X_train[:, 2]
                 + rng_ltv.normal(0, 200, len(X_train)))
    ltv_model = GradientBoostingRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.1, random_state=0)
    ltv_model.fit(X_train, ltv_proxy)
    ltv_hat = ltv_model.predict(X_test).clip(50, 8000)

    # Churn: binary control non-retention as proxy
    ctrl_mask = W_train == 0
    churn_proxy = 1 - Y_train[ctrl_mask]
    churn_model = GradientBoostingClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.1, random_state=0)
    churn_model.fit(X_train[ctrl_mask], churn_proxy)
    churn_hat = churn_model.predict_proba(X_test)[:, 1].clip(0.01, 0.99)

    return ltv_hat, churn_hat


def _estimate_cate_tlearner(X_train, X_test, W_train, Y_train):
    """T-Learner CATE estimation."""
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
    return tau_hat


def _estimate_cate_xlearner(X_train, X_test, W_train, Y_train):
    """X-Learner CATE estimation (Kunzel et al. 2019)."""
    treat_mask = W_train == 1
    ctrl_mask = W_train == 0

    # Step 1: Base response models (same as T-Learner)
    mu1 = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0)
    mu0 = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0)
    mu1.fit(X_train[treat_mask], Y_train[treat_mask])
    mu0.fit(X_train[ctrl_mask], Y_train[ctrl_mask])

    # Step 2: Imputed treatment effects
    D1 = Y_train[treat_mask] - mu0.predict_proba(X_train[treat_mask])[:, 1]
    D0 = mu1.predict_proba(X_train[ctrl_mask])[:, 1] - Y_train[ctrl_mask]

    # Step 3: CATE models on imputed effects
    tau1_model = GradientBoostingRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.1, random_state=0)
    tau0_model = GradientBoostingRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.1, random_state=0)
    tau1_model.fit(X_train[treat_mask], D1)
    tau0_model.fit(X_train[ctrl_mask], D0)

    # Step 4: Combine with known propensity e(x) = 0.5
    tau_hat = 0.5 * tau0_model.predict(X_test) + 0.5 * tau1_model.predict(X_test)
    return tau_hat


def estimate_components(data, train_idx, test_idx, cate_method='tlearner'):
    """Estimate CATE, LTV, and churn from training data."""
    X = data['features']
    W = data['treatment']
    Y = data['y_obs']
    X_train, X_test = X[train_idx], X[test_idx]
    W_train, Y_train = W[train_idx], Y[train_idx]

    if cate_method == 'xlearner':
        tau_hat = _estimate_cate_xlearner(X_train, X_test, W_train, Y_train)
    else:
        tau_hat = _estimate_cate_tlearner(X_train, X_test, W_train, Y_train)

    ltv_hat, churn_hat = _estimate_ltv_churn(X_train, X_test, W_train, Y_train, train_idx)
    return tau_hat, ltv_hat, churn_hat


def estimate_components_misspec(data, train_idx, test_idx, cate_method='tlearner'):
    """Estimate with MISSPECIFIED CATE model: login_freq dropped."""
    X = data['features']
    W = data['treatment']
    Y = data['y_obs']
    X_train, X_test = X[train_idx], X[test_idx]
    W_train, Y_train = W[train_idx], Y[train_idx]

    # CATE model uses only 3 features (drops login_freq at index 2)
    cate_cols = [0, 1, 3]
    X_train_cate = X_train[:, cate_cols]
    X_test_cate = X_test[:, cate_cols]

    if cate_method == 'xlearner':
        tau_hat = _estimate_cate_xlearner(X_train_cate, X_test_cate, W_train, Y_train)
    else:
        tau_hat = _estimate_cate_tlearner(X_train_cate, X_test_cate, W_train, Y_train)

    # LTV and churn use ALL features
    ltv_hat, churn_hat = _estimate_ltv_churn(X_train, X_test, W_train, Y_train, train_idx)
    return tau_hat, ltv_hat, churn_hat


def _estimate_slearner(X_train, X_test, W_train, Y_train, ltv_hat):
    """S-Learner: single model with treatment as feature [C3]."""
    X_aug = np.column_stack([X_train, W_train])
    slearner = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=0)
    slearner.fit(X_aug, Y_train)
    n_test = len(X_test)
    pred_t1 = slearner.predict(np.column_stack([X_test, np.ones(n_test)]))
    pred_t0 = slearner.predict(np.column_stack([X_test, np.zeros(n_test)]))
    tau_sl = pred_t1 - pred_t0
    return tau_sl * ltv_hat


# --- Scoring and evaluation ---

def compute_welfare(targeted_idx, true_tau, true_ltv):
    return float(np.sum(true_tau[targeted_idx] * true_ltv[targeted_idx]))


def compute_risk_exposure(targeted_idx, true_churn):
    return float(np.mean(true_churn[targeted_idx]))


def compute_mean_churn_moderation(targeted_idx, true_churn):
    return float(np.mean(4.0 * true_churn[targeted_idx] * (1.0 - true_churn[targeted_idx])))


def target_topk(scores, k):
    return np.argsort(scores)[-k:]


# --- Expanded calibration grid [M15] ---
THETA_GRID = [
    # g~1 identity (allows calibration to select no risk modulation)
    {'pL': 0.00, 'pML': 0.01, 'pMH': 0.99, 'phi': 1.0},
    # Original configs
    {'pL': 0.03, 'pML': 0.30, 'pMH': 0.65, 'phi': 0.15},
    {'pL': 0.05, 'pML': 0.35, 'pMH': 0.60, 'phi': 0.20},
    {'pL': 0.05, 'pML': 0.40, 'pMH': 0.60, 'phi': 0.20},  # default
    {'pL': 0.08, 'pML': 0.30, 'pMH': 0.55, 'phi': 0.10},
    {'pL': 0.10, 'pML': 0.35, 'pMH': 0.55, 'phi': 0.15},
    {'pL': 0.05, 'pML': 0.25, 'pMH': 0.50, 'phi': 0.10},
    {'pL': 0.03, 'pML': 0.20, 'pMH': 0.55, 'phi': 0.10},
    {'pL': 0.05, 'pML': 0.30, 'pMH': 0.55, 'phi': 0.15},
    {'pL': 0.08, 'pML': 0.25, 'pMH': 0.50, 'phi': 0.05},
    # Extended range (not aligned with inverse-U)
    {'pL': 0.00, 'pML': 0.10, 'pMH': 0.90, 'phi': 0.50},
    {'pL': 0.15, 'pML': 0.40, 'pMH': 0.50, 'phi': 0.00},
    {'pL': 0.00, 'pML': 0.05, 'pMH': 0.40, 'phi': 0.30},
    {'pL': 0.20, 'pML': 0.50, 'pMH': 0.80, 'phi': 0.40},
    {'pL': 0.10, 'pML': 0.20, 'pMH': 0.40, 'phi': 0.05},
    {'pL': 0.05, 'pML': 0.15, 'pMH': 0.70, 'phi': 0.30},
    {'pL': 0.00, 'pML': 0.30, 'pMH': 0.70, 'phi': 0.80},
    {'pL': 0.02, 'pML': 0.25, 'pMH': 0.45, 'phi': 0.10},
    {'pL': 0.05, 'pML': 0.45, 'pMH': 0.55, 'phi': 0.15},
    {'pL': 0.10, 'pML': 0.30, 'pMH': 0.60, 'phi': 0.25},
    {'pL': 0.03, 'pML': 0.35, 'pMH': 0.65, 'phi': 0.05},
    {'pL': 0.00, 'pML': 0.50, 'pMH': 0.80, 'phi': 0.60},
]
DEFAULT_THETA_IDX = 3  # pL=0.05, pML=0.40, pMH=0.60, phi=0.20


def calibrate_theta(data, train_idx, seed, k, est_func=None, cate_method='tlearner'):
    """Calibrate theta by held-out validation with grid search."""
    if est_func is None:
        est_func = estimate_components
    est_rel, val_rel = train_test_split(
        np.arange(len(train_idx)), test_size=0.40, random_state=seed + 99)
    est_idx = train_idx[est_rel]
    val_idx = train_idx[val_rel]

    tau_val, ltv_val, churn_val = est_func(data, est_idx, val_idx,
                                           cate_method=cate_method)
    k_val = int(len(val_idx) * DEFAULT_BUDGET)

    best_welfare = -np.inf
    best_config = THETA_GRID[DEFAULT_THETA_IDX]
    for cfg in THETA_GRID:
        g_val = f_churn_vec(churn_val, **cfg)
        scores = tau_val * ltv_val * g_val
        targeted = target_topk(scores, k_val)
        w = float(np.sum(tau_val[targeted] * ltv_val[targeted]))
        if w > best_welfare:
            best_welfare = w
            best_config = cfg
    return best_config


# --- Statistical testing utilities [M3] ---

def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction to a list of p-values."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted = np.ones(n)
    for rank, idx in enumerate(sorted_indices):
        adjusted[idx] = min(1.0, p_values[idx] * (n - rank))
    for i in range(1, n):
        idx = sorted_indices[i]
        prev_idx = sorted_indices[i - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return adjusted


def tost_paired(x, y, margin):
    """Two one-sided test for equivalence with margin."""
    diff = np.array(x) - np.array(y)
    n = len(diff)
    mean_diff = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    t_upper = (mean_diff - margin) / se
    t_lower = (mean_diff + margin) / se
    p_upper = scipy_stats.t.cdf(t_upper, df=n - 1)
    p_lower = 1 - scipy_stats.t.cdf(t_lower, df=n - 1)
    p_tost = max(p_upper, p_lower)
    return {
        'mean_diff': round(float(mean_diff), 0),
        'margin': round(float(margin), 0),
        'p_tost': round(float(p_tost), 4),
        'equivalent': bool(p_tost < 0.05),
    }


def cohens_d_paired(x, y):
    """Cohen's d for paired observations."""
    diff = np.array(x) - np.array(y)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return round(float(np.mean(diff) / sd), 3)


def ci_95(x, y):
    """95% CI for mean difference of paired observations."""
    diff = np.array(x) - np.array(y)
    n = len(diff)
    mean_d = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    margin = scipy_stats.t.ppf(0.975, df=n - 1) * se
    return (round(float(mean_d - margin), 0), round(float(mean_d + margin), 0))


# --- Main simulation loop ---

def run_single_simulation(seed, budget_frac=DEFAULT_BUDGET, dgp_func=None,
                          est_func=None, cate_method='tlearner'):
    if dgp_func is None:
        dgp_func = generate_population
    if est_func is None:
        est_func = estimate_components
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

    tau_hat, ltv_hat, churn_hat = est_func(data, train_idx, test_idx,
                                           cate_method=cate_method)

    g_hat = f_churn_vec(churn_hat)
    g_oracle = f_churn_vec(true_churn)

    cal_config = calibrate_theta(data, train_idx, seed, k,
                                 est_func=est_func, cate_method=cate_method)
    g_cal = f_churn_vec(churn_hat, **cal_config)

    strategies = {}
    strategies['Random'] = rng.uniform(0, 1, len(test_idx))
    strategies['CATE Only'] = tau_hat
    strategies['CATE x LTV'] = tau_hat * ltv_hat
    strategies['Full Framework (Est.)'] = tau_hat * ltv_hat * g_hat
    strategies['Full Framework (Cal.)'] = tau_hat * ltv_hat * g_cal
    strategies['Full Framework (Oracle)'] = (data['true_tau'][test_idx]
                                              * data['true_ltv'][test_idx]
                                              * g_oracle)

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


def _aggregate_results(all_results, strategy_names):
    """Aggregate per-run results into summary statistics."""
    agg = {}
    for name in strategy_names:
        runs = all_results[name]
        agg[name] = {
            'ate_mean': round(np.mean([r['ate'] for r in runs]), 1),
            'ate_sd': round(np.std([r['ate'] for r in runs]), 1),
            'welfare_mean': round(np.mean([r['welfare'] for r in runs]), 0),
            'welfare_sd': round(np.std([r['welfare'] for r in runs]), 0),
            'risk_mean': round(np.mean([r['risk_exposure'] for r in runs]), 3),
            'risk_sd': round(np.std([r['risk_exposure'] for r in runs]), 3),
            'mean_churn_mod': round(np.mean([r['mean_churn_mod'] for r in runs]), 3),
            'mean_ltv_cr': round(np.mean([r['mean_ltv_cr'] for r in runs]), 0),
        }
    return agg


def _paired_test(w1, w2, label):
    """Compute paired t-test with effect size and CI."""
    diff = np.array(w1) - np.array(w2)
    t_stat, p_val = scipy_stats.ttest_rel(w1, w2)
    d = cohens_d_paired(w1, w2)
    lo, hi = ci_95(w1, w2)
    return {
        'mean_diff': round(float(np.mean(diff)), 0),
        'se_diff': round(float(np.std(diff) / np.sqrt(len(diff))), 0),
        't_stat': round(float(t_stat), 2),
        'p_value': round(float(p_val), 4),
        'cohens_d': d,
        'ci_95': [lo, hi],
    }


def main():
    print("=" * 70)
    print("SIMULATION STUDY v29: Risk-Modulated Targeting")
    print("=" * 70)
    print(f"N={N_PLAYERS}, Runs={N_RUNS}, Default budget={DEFAULT_BUDGET*100:.0f}%")
    print(f"Calibration grid: {len(THETA_GRID)} configs (including g~1 identity)")

    strategy_names = ['Random', 'CATE Only', 'CATE x LTV',
                      'Full Framework (Est.)', 'Full Framework (Cal.)',
                      'Full Framework (Oracle)', 'S-Learner']

    output = {}

    # =====================================================================
    # 1. Main DGP + T-Learner
    # =====================================================================
    print("\n--- Scenario 1: Main DGP, T-Learner ---")
    all_results = {s: [] for s in strategy_names}
    all_frontiers = []
    all_misspec = []
    all_cal_configs = []

    for run in range(N_RUNS):
        if (run + 1) % 50 == 0:
            print(f"  Run {run + 1}/{N_RUNS}...")
        seed = SEED_BASE + run * 7
        results, frontier, misspec_inv, cal_cfg = run_single_simulation(seed)
        for name, res in results.items():
            all_results[name].append(res)
        all_frontiers.append(frontier)
        all_misspec.append(misspec_inv)
        all_cal_configs.append(cal_cfg)

    output['main_results'] = _aggregate_results(all_results, strategy_names)

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
        ('Full Framework (Oracle)', 'CATE x LTV'),
    ]
    paired_tests = {}
    raw_pvalues = []
    for s1, s2 in comparisons:
        w1 = [r['welfare'] for r in all_results[s1]]
        w2 = [r['welfare'] for r in all_results[s2]]
        label = f'{s1} vs {s2}'
        paired_tests[label] = _paired_test(w1, w2, label)
        raw_pvalues.append(paired_tests[label]['p_value'])

    corrected = holm_bonferroni(raw_pvalues)
    for i, (s1, s2) in enumerate(comparisons):
        label = f'{s1} vs {s2}'
        paired_tests[label]['p_holm'] = round(float(corrected[i]), 4)
    output['paired_tests'] = paired_tests

    # =====================================================================
    # 2. Main DGP + X-Learner (robustness)
    # =====================================================================
    print("\n--- Scenario 2: Main DGP, X-Learner ---")
    xl_results = {s: [] for s in strategy_names}
    for run in range(N_RUNS):
        if (run + 1) % 50 == 0:
            print(f"  Run {run + 1}/{N_RUNS}...")
        seed = SEED_BASE + run * 7
        results, _, _, _ = run_single_simulation(seed, cate_method='xlearner')
        for name, res in results.items():
            xl_results[name].append(res)
    output['xlearner_results'] = _aggregate_results(xl_results, strategy_names)

    # Key paired test for X-Learner
    w_cal_xl = [r['welfare'] for r in xl_results['Full Framework (Cal.)']]
    w_cate_xl = [r['welfare'] for r in xl_results['CATE Only']]
    w_ltv_xl = [r['welfare'] for r in xl_results['CATE x LTV']]
    output['xlearner_paired'] = {
        'cal_vs_cate_only': _paired_test(w_cal_xl, w_cate_xl, 'XL cal vs cate'),
        'cal_vs_ltv': _paired_test(w_cal_xl, w_ltv_xl, 'XL cal vs ltv'),
    }

    # =====================================================================
    # 3. CATE Misspecification + T-Learner
    # =====================================================================
    print("\n--- Scenario 3: CATE Misspecification, T-Learner ---")
    misspec_strats = ['CATE Only', 'CATE x LTV',
                      'Full Framework (Est.)', 'Full Framework (Cal.)']
    misspec_results = {s: [] for s in misspec_strats}
    for run in range(N_RUNS):
        if (run + 1) % 50 == 0:
            print(f"  Run {run + 1}/{N_RUNS}...")
        seed = SEED_BASE + run * 7
        results, _, _, _ = run_single_simulation(
            seed, est_func=estimate_components_misspec)
        for name in misspec_strats:
            misspec_results[name].append(results[name])

    output['cate_misspec'] = {}
    for name in misspec_strats:
        runs = misspec_results[name]
        output['cate_misspec'][name] = {
            'welfare_mean': round(np.mean([r['welfare'] for r in runs]), 0),
            'welfare_sd': round(np.std([r['welfare'] for r in runs]), 0),
            'ate_mean': round(np.mean([r['ate'] for r in runs]), 1),
        }

    # Paired tests under misspecification
    w_cal_ms = [r['welfare'] for r in misspec_results['Full Framework (Cal.)']]
    w_cate_ms = [r['welfare'] for r in misspec_results['CATE Only']]
    w_ltv_ms = [r['welfare'] for r in misspec_results['CATE x LTV']]

    ms_paired = {
        'cal_vs_cate': _paired_test(w_cal_ms, w_cate_ms, 'ms cal vs cate'),
        'cal_vs_ltv': _paired_test(w_cal_ms, w_ltv_ms, 'ms cal vs ltv'),
    }
    ms_raw_p = [ms_paired['cal_vs_cate']['p_value'],
                ms_paired['cal_vs_ltv']['p_value']]
    ms_corrected = holm_bonferroni(ms_raw_p)
    ms_paired['cal_vs_cate']['p_holm'] = round(float(ms_corrected[0]), 4)
    ms_paired['cal_vs_ltv']['p_holm'] = round(float(ms_corrected[1]), 4)

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
    for run in range(N_RUNS):
        if (run + 1) % 50 == 0:
            print(f"  Run {run + 1}/{N_RUNS}...")
        seed = SEED_BASE + run * 7
        results, _, _, _ = run_single_simulation(
            seed, est_func=estimate_components_misspec, cate_method='xlearner')
        for name in misspec_strats:
            xl_misspec_results[name].append(results[name])

    output['xlearner_misspec'] = {}
    for name in misspec_strats:
        runs = xl_misspec_results[name]
        output['xlearner_misspec'][name] = {
            'welfare_mean': round(np.mean([r['welfare'] for r in runs]), 0),
            'welfare_sd': round(np.std([r['welfare'] for r in runs]), 0),
            'ate_mean': round(np.mean([r['ate'] for r in runs]), 1),
        }
    w_cal_xlm = [r['welfare'] for r in xl_misspec_results['Full Framework (Cal.)']]
    w_cate_xlm = [r['welfare'] for r in xl_misspec_results['CATE Only']]
    output['xlearner_misspec_paired'] = {
        'cal_vs_cate': _paired_test(w_cal_xlm, w_cate_xlm, 'xlm cal vs cate'),
    }

    # =====================================================================
    # 5. Null DGP
    # =====================================================================
    print("\n--- Scenario 5: Null DGP (no CATE-churn interaction) ---")
    null_strats = ['CATE Only', 'CATE x LTV',
                   'Full Framework (Est.)', 'Full Framework (Cal.)']
    null_results = {s: [] for s in null_strats}
    for run in range(N_RUNS):
        if (run + 1) % 50 == 0:
            print(f"  Run {run + 1}/{N_RUNS}...")
        seed = SEED_BASE + run * 7
        results, _, _, _ = run_single_simulation(
            seed, dgp_func=generate_population_null)
        for name in null_strats:
            null_results[name].append(results[name])

    output['null_dgp'] = {}
    for name in null_strats:
        runs = null_results[name]
        output['null_dgp'][name] = {
            'welfare_mean': round(np.mean([r['welfare'] for r in runs]), 0),
            'welfare_sd': round(np.std([r['welfare'] for r in runs]), 0),
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
    for run in range(N_RUNS):
        if (run + 1) % 50 == 0:
            print(f"  Run {run + 1}/{N_RUNS}...")
        seed = SEED_BASE + run * 7
        results, _, _, _ = run_single_simulation(
            seed, dgp_func=generate_population_correlated)
        for name in corr_strats:
            corr_results[name].append(results[name])

    output['correlated_dgp'] = {}
    for name in corr_strats:
        runs = corr_results[name]
        output['correlated_dgp'][name] = {
            'welfare_mean': round(np.mean([r['welfare'] for r in runs]), 0),
            'welfare_sd': round(np.std([r['welfare'] for r in runs]), 0),
            'ate_mean': round(np.mean([r['ate'] for r in runs]), 1),
        }
    w_cal_corr = [r['welfare'] for r in corr_results['Full Framework (Cal.)']]
    w_cate_corr = [r['welfare'] for r in corr_results['CATE Only']]
    w_ltv_corr = [r['welfare'] for r in corr_results['CATE x LTV']]
    output['correlated_dgp_paired'] = {
        'cal_vs_cate': _paired_test(w_cal_corr, w_cate_corr, 'corr cal vs cate'),
        'cal_vs_ltv': _paired_test(w_cal_corr, w_ltv_corr, 'corr cal vs ltv'),
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
        for run in range(N_RUNS):
            seed = SEED_BASE + run * 7
            results, _, _, _ = run_single_simulation(seed, budget_frac=kappa)
            for name, res in results.items():
                budget_results[str(kappa)][name].append(res)

    output['budget_sensitivity'] = {}
    for kappa_str, strat_runs in budget_results.items():
        output['budget_sensitivity'][kappa_str] = {}
        for name in strategy_names:
            if strat_runs[name]:
                output['budget_sensitivity'][kappa_str][name] = {
                    'welfare_mean': round(np.mean([r['welfare'] for r in strat_runs[name]]), 0),
                    'ate_mean': round(np.mean([r['ate'] for r in strat_runs[name]]), 1),
                }

    # Budget paired tests with Holm-Bonferroni
    for kappa_str, strat_runs in budget_results.items():
        w_cal = [r['welfare'] for r in strat_runs['Full Framework (Cal.)']]
        w_ltv = [r['welfare'] for r in strat_runs['CATE x LTV']]
        w_cate = [r['welfare'] for r in strat_runs['CATE Only']]
        bp = {
            'cal_vs_ltv': _paired_test(w_cal, w_ltv, f'{kappa_str} cal vs ltv'),
            'cal_vs_cate_only': _paired_test(w_cal, w_cate, f'{kappa_str} cal vs cate'),
        }
        raw_p = [bp['cal_vs_ltv']['p_value'], bp['cal_vs_cate_only']['p_value']]
        corrected = holm_bonferroni(raw_p)
        bp['cal_vs_ltv']['p_holm'] = round(float(corrected[0]), 4)
        bp['cal_vs_cate_only']['p_holm'] = round(float(corrected[1]), 4)
        budget_paired[kappa_str] = bp
    output['budget_paired_tests'] = budget_paired

    # =====================================================================
    # Metadata
    # =====================================================================
    output['dgp'] = DGP_SPEC
    output['design'] = {
        'n_players': N_PLAYERS, 'n_runs': N_RUNS,
        'default_budget': DEFAULT_BUDGET, 'treatment_prob': TREATMENT_PROB,
        'budget_sensitivity_runs': N_RUNS,
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

    # Save (with numpy-safe encoder)
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    output_path = os.path.join(os.path.dirname(__file__), '..',
                               'outputs', 'simulation_v29.json')
    with open(output_path, 'w') as fp:
        json.dump(output, fp, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_path}")


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


if __name__ == '__main__':
    verify_population_moments()
    main()
