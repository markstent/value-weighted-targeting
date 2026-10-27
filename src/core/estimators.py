"""CATE, LTV and churn estimators plus held-out theta calibration."""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from core.policy import (DEFAULT_BUDGET, DEFAULT_THETA_IDX, THETA_GRID,
                         f_churn_vec, target_topk)

# Independent RNG streams per replication seed (see seed_from).
LTV_STREAM = 0        # LTV-proxy noise for the deployed estimator
CALIB_STREAM = 1      # LTV-proxy noise for the calibration (validation) fit
VALUE_STREAM_V41 = 2  # end-to-end joint value model (compute_v41)
VALUE_STREAM_V42 = 3  # well-specified value model (compute_v42)
VALUE_STREAM_V43 = 4  # richer reference value model (compute_v43)


def seed_from(seed, stream):
    """Deterministic child seed for (replication seed, stream); replaces hash()."""
    return (seed * 1_000_003 + stream) % 2**31


def _estimate_ltv_churn(X_train, X_test, W_train, Y_train, seed, stream=LTV_STREAM):
    """Shared LTV and churn estimation (used by all CATE methods).

    The LTV-proxy noise is seeded by seed_from(seed, stream), where seed is the
    replication seed."""
    # LTV: misspecified proxy (drops age, nonlinear spend, different coefficients)
    rng_ltv = np.random.RandomState(seed_from(seed, stream))
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


def estimate_components(data, train_idx, test_idx, cate_method='tlearner', *,
                        seed, stream=LTV_STREAM):
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

    ltv_hat, churn_hat = _estimate_ltv_churn(X_train, X_test, W_train, Y_train,
                                              seed, stream)
    return tau_hat, ltv_hat, churn_hat


def estimate_components_misspec(data, train_idx, test_idx, cate_method='tlearner', *,
                                seed, stream=LTV_STREAM):
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
    ltv_hat, churn_hat = _estimate_ltv_churn(X_train, X_test, W_train, Y_train,
                                              seed, stream)
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



def calibrate_theta(data, train_idx, seed, k, est_func=None, cate_method='tlearner'):
    """Calibrate theta by held-out validation with grid search."""
    if est_func is None:
        est_func = estimate_components
    est_rel, val_rel = train_test_split(
        np.arange(len(train_idx)), test_size=0.40, random_state=seed + 99)
    est_idx = train_idx[est_rel]
    val_idx = train_idx[val_rel]

    tau_val, ltv_val, churn_val = est_func(data, est_idx, val_idx,
                                           cate_method=cate_method,
                                           seed=seed, stream=CALIB_STREAM)
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
