"""Targeting policy primitives: risk modulation g, top-k selection, scoring."""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, train_test_split

DEFAULT_BUDGET = 0.30


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


def compute_welfare(targeted_idx, true_tau, true_ltv):
    return float(np.sum(true_tau[targeted_idx] * true_ltv[targeted_idx]))


def compute_risk_exposure(targeted_idx, true_churn):
    return float(np.mean(true_churn[targeted_idx]))


def compute_mean_churn_moderation(targeted_idx, true_churn):
    return float(np.mean(4.0 * true_churn[targeted_idx] * (1.0 - true_churn[targeted_idx])))


def target_topk(scores, k):
    return np.argsort(scores)[-k:]


def h_protection(harm, h_lo=0.60, h_hi=0.90):
    """Harm-protection weight: 1 below h_lo, ramps to 0 above h_hi."""
    h = np.where(harm <= h_lo, 1.0,
                 np.where(harm >= h_hi, 0.0,
                          1.0 - (harm - h_lo) / (h_hi - h_lo)))
    return h


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


PROPENSITY = 0.5  # known randomised assignment probability


def _crossfit_aipw_scores(X, W, Y, seed, n_folds, e=PROPENSITY):
    """Doubly-robust scores Gamma_i on a fold, with cross-fitted outcome models.

    Gamma = mu1 - mu0 + W (Y - mu1) / e - (1 - W) (Y - mu0) / (1 - e);
    the propensity e is known, so only mu0 and mu1 are estimated.  Each unit's
    mu1, mu0 come from models fit on the other folds."""
    from core.estimators import CROSSFIT_STREAM, seed_from
    n = len(Y)
    m1 = np.empty(n)
    m0 = np.empty(n)
    kf = KFold(n_splits=n_folds, shuffle=True,
               random_state=seed_from(seed, CROSSFIT_STREAM))
    for fit, pred in kf.split(np.arange(n)):
        for arm, out in ((1, m1), (0, m0)):
            sel = fit[W[fit] == arm]
            y_arm = Y[sel]
            if len(np.unique(y_arm)) < 2:   # degenerate arm: constant outcome model
                out[pred] = float(y_arm.mean()) if len(y_arm) else 0.5
                continue
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0)
            model.fit(X[sel], y_arm)
            out[pred] = model.predict_proba(X[pred])[:, 1]
    return (m1 - m0) + W / e * (Y - m1) - (1 - W) / (1 - e) * (Y - m0)


def calibrate_theta(data, train_idx, seed, kappa, objective="aipw", n_folds=2,
                    est_func=None, cate_method='tlearner'):
    """Calibrate theta on a validation fold inside the training split, at budget kappa.

    objective="aipw": policy value of theta is the sum over its selected top-k
      units of Gamma_i * ltv_hat_i, with Gamma the cross-fitted doubly-robust
      score (known propensity) and ltv_hat the estimated value (V is not
      observable).
    objective="plugin": sum of tau_hat * ltv_hat over the selected units.  This
      is maximised by g=1 for any estimates (top-k tautology) and is kept only
      to demonstrate that.
    Returns the argmax config of THETA_GRID (ties go to the lowest index)."""
    if objective not in ("aipw", "plugin"):
        raise ValueError(f"unknown objective {objective!r}")
    from core.estimators import CALIB_STREAM, estimate_components
    if est_func is None:
        est_func = estimate_components
    est_rel, val_rel = train_test_split(
        np.arange(len(train_idx)), test_size=0.40, random_state=seed + 99)
    est_idx = train_idx[est_rel]
    val_idx = train_idx[val_rel]

    tau_val, ltv_val, churn_val = est_func(data, est_idx, val_idx,
                                           cate_method=cate_method,
                                           seed=seed, stream=CALIB_STREAM)
    k_val = int(len(val_idx) * kappa)

    if objective == "aipw":
        gamma = _crossfit_aipw_scores(data['features'][val_idx],
                                      data['treatment'][val_idx],
                                      data['y_obs'][val_idx], seed, n_folds)
        unit_value = gamma * ltv_val
    else:
        unit_value = tau_val * ltv_val

    best_value = -np.inf
    best_config = THETA_GRID[0]
    for cfg in THETA_GRID:
        g_val = f_churn_vec(churn_val, **cfg)
        targeted = np.sort(target_topk(tau_val * ltv_val * g_val, k_val))
        w = float(np.sum(unit_value[targeted]))
        if w > best_value:      # strict: ties keep the lowest grid index
            best_value = w
            best_config = cfg
    return best_config
