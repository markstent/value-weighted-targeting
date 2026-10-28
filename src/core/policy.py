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
    h = np.where(harm <= h_lo, 1.0, np.where(harm >= h_hi, 0.0, 1.0 - (harm - h_lo) / (h_hi - h_lo)))
    return h


# --- Expanded calibration grid [M15] ---
THETA_GRID = [
    # g~1 identity (allows calibration to select no risk modulation)
    {"pL": 0.00, "pML": 0.01, "pMH": 0.99, "phi": 1.0},
    # Original configs
    {"pL": 0.03, "pML": 0.30, "pMH": 0.65, "phi": 0.15},
    {"pL": 0.05, "pML": 0.35, "pMH": 0.60, "phi": 0.20},
    {"pL": 0.05, "pML": 0.40, "pMH": 0.60, "phi": 0.20},  # default
    {"pL": 0.08, "pML": 0.30, "pMH": 0.55, "phi": 0.10},
    {"pL": 0.10, "pML": 0.35, "pMH": 0.55, "phi": 0.15},
    {"pL": 0.05, "pML": 0.25, "pMH": 0.50, "phi": 0.10},
    {"pL": 0.03, "pML": 0.20, "pMH": 0.55, "phi": 0.10},
    {"pL": 0.05, "pML": 0.30, "pMH": 0.55, "phi": 0.15},
    {"pL": 0.08, "pML": 0.25, "pMH": 0.50, "phi": 0.05},
    # Extended range (not aligned with inverse-U)
    {"pL": 0.00, "pML": 0.10, "pMH": 0.90, "phi": 0.50},
    {"pL": 0.15, "pML": 0.40, "pMH": 0.50, "phi": 0.00},
    {"pL": 0.00, "pML": 0.05, "pMH": 0.40, "phi": 0.30},
    {"pL": 0.20, "pML": 0.50, "pMH": 0.80, "phi": 0.40},
    {"pL": 0.10, "pML": 0.20, "pMH": 0.40, "phi": 0.05},
    {"pL": 0.05, "pML": 0.15, "pMH": 0.70, "phi": 0.30},
    {"pL": 0.00, "pML": 0.30, "pMH": 0.70, "phi": 0.80},
    {"pL": 0.02, "pML": 0.25, "pMH": 0.45, "phi": 0.10},
    {"pL": 0.05, "pML": 0.45, "pMH": 0.55, "phi": 0.15},
    {"pL": 0.10, "pML": 0.30, "pMH": 0.60, "phi": 0.25},
    {"pL": 0.03, "pML": 0.35, "pMH": 0.65, "phi": 0.05},
    {"pL": 0.00, "pML": 0.50, "pMH": 0.80, "phi": 0.60},
]
DEFAULT_THETA_IDX = 3  # pL=0.05, pML=0.40, pMH=0.60, phi=0.20


PROPENSITY = 0.5  # known randomised assignment probability


def aipw_scores(X, W, Y, seed, n_folds=2, e=PROPENSITY, params=None):
    """Doubly-robust scores Gamma_i on a fold, with cross-fitted outcome models.

    Gamma = mu1 - mu0 + W (Y - mu1) / e - (1 - W) (Y - mu0) / (1 - e);
    the propensity e is known, so only mu0 and mu1 are estimated.  Each unit's
    mu1, mu0 come from models fit on the other folds.  params: optional
    {max_depth, n_estimators, learning_rate} for the outcome models (default
    depth 3, 100 trees, rate 0.1)."""
    from core.estimators import CROSSFIT_STREAM, seed_from

    n = len(Y)
    m1 = np.empty(n)
    m0 = np.empty(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed_from(seed, CROSSFIT_STREAM))
    for fit, pred in kf.split(np.arange(n)):
        for arm, out in ((1, m1), (0, m0)):
            sel = fit[W[fit] == arm]
            y_arm = Y[sel]
            if len(np.unique(y_arm)) < 2:  # degenerate arm: constant outcome model
                out[pred] = float(y_arm.mean()) if len(y_arm) else 0.5
                continue
            kw = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1} if params is None else params
            model = GradientBoostingClassifier(random_state=0, **kw)
            model.fit(X[sel], y_arm)
            out[pred] = model.predict_proba(X[pred])[:, 1]
    return (m1 - m0) + W / e * (Y - m1) - (1 - W) / (1 - e) * (Y - m0)


def calibrate_theta_all(
    data, train_idx, seed, kappa, objectives=("aipw", "plugin"), n_folds=2, est_func=None, cate_method="tlearner"
):
    """Calibrate theta for several objectives, fitting the est->val components once.

    Returns {objective: config}.  objective="aipw": policy value of theta is
      the sum over its selected top-k units of Gamma_i * ltv_hat_i, with Gamma
      the cross-fitted doubly-robust score (known propensity) and ltv_hat the
      estimated value (V is not observable).
    objective="plugin": sum of tau_hat * ltv_hat over the selected units.  This
      is maximised by g=1 for any estimates (top-k tautology) and is kept only
      to demonstrate that.
    Selection is the argmax over THETA_GRID at k = int(n_val * kappa); ties go
    to the lowest grid index."""
    for objective in objectives:
        if objective not in ("aipw", "plugin"):
            raise ValueError(f"unknown objective {objective!r}")
    from core.estimators import CALIB_SPLIT_STREAM, CALIB_STREAM, estimate_components, seed_from

    if est_func is None:
        est_func = estimate_components
    est_rel, val_rel = train_test_split(
        np.arange(len(train_idx)), test_size=0.40, random_state=seed_from(seed, CALIB_SPLIT_STREAM)
    )
    est_idx = train_idx[est_rel]
    val_idx = train_idx[val_rel]

    tau_val, ltv_val, churn_val = est_func(
        data, est_idx, val_idx, cate_method=cate_method, seed=seed, stream=CALIB_STREAM
    )
    k_val = int(len(val_idx) * kappa)
    base = tau_val * ltv_val

    out = {}
    for objective in objectives:
        if objective == "aipw":
            gamma = aipw_scores(
                data["features"][val_idx], data["treatment"][val_idx], data["y_obs"][val_idx], seed, n_folds
            )
            unit_value = gamma * ltv_val
        else:
            unit_value = base
        best_value = -np.inf
        best_config = THETA_GRID[0]
        for cfg in THETA_GRID:
            g_val = f_churn_vec(churn_val, **cfg)
            targeted = np.sort(target_topk(base * g_val, k_val))
            w = float(np.sum(unit_value[targeted]))
            if w > best_value:  # strict: ties keep the lowest grid index
                best_value = w
                best_config = cfg
        out[objective] = best_config
    return out


def calibrate_theta(data, train_idx, seed, kappa, objective="aipw", n_folds=2, est_func=None, cate_method="tlearner"):
    """Single-objective calibration; see calibrate_theta_all."""
    return calibrate_theta_all(data, train_idx, seed, kappa, (objective,), n_folds, est_func, cate_method)[objective]


def misrank_terms(tau_hat, v_hat, tau_ref, v_ref, k):
    """Decomposition of a BOUND on the mean absolute score gap over the misranked set (None
    if the set is empty).  This is NOT a regret decomposition: it says nothing about welfare
    lost, only how the score error splits between the two components.

    M = top-k by the deployed score tau_hat * v_hat XOR top-k by the reference score
    tau_ref * v_ref.  For every unit the score gap telescopes EXACTLY in one fixed order:

        tau_ref v_ref - tau_hat v_hat = (tau_ref - tau_hat) v_ref + tau_hat (v_ref - v_hat)

    (CATE error evaluated at the REFERENCE value, then value error at the DEPLOYED CATE).
    The terms are the means over M of the absolute pieces, |tau_ref - tau_hat| |v_ref|
    (CATE term) and |tau_hat| |v_ref - v_hat| (value term).  By the triangle inequality

        mean_M |tau_ref v_ref - tau_hat v_hat| <= CATE term + VALUE term,

    with equality when the two signed pieces have the same sign on every unit of M (always
    when only one component is in error).  A "CATE share" built from the terms is a share of
    this bound, in this fixed order (the order matters: the other telescoping order gives
    different terms).  Returns (cate_term, value_term, |M|)."""
    pi_dep = set(target_topk(tau_hat * v_hat, k).tolist())
    pi_ref = set(target_topk(tau_ref * v_ref, k).tolist())
    m = np.array(sorted(pi_dep.symmetric_difference(pi_ref)), dtype=int)
    if len(m) == 0:
        return None
    return (
        float((np.abs(tau_ref[m] - tau_hat[m]) * np.abs(v_ref[m])).mean()),
        float((np.abs(tau_hat[m]) * np.abs(v_ref[m] - v_hat[m])).mean()),
        len(m),
    )


def regret_split(tau_ref, ltv_ref, tau_hat, ltv_hat, tau_true, ltv_true, k):
    """Order-free split of welfare regret into CATE error and value-model error.

    W(a, b) is the realised TRUE welfare (tau_true, ltv_true) of the top-k set by
    the score a*b.  R(a, b) = W(tau_ref, ltv_ref) - W(a, b).  The reference is the
    feasible oracle (tau_bar, ltv_bar), so R(ref, ref) = 0 by construction and the
    total regret is R(tau_hat, ltv_hat).

    Two single-path telescoping splits, each assigning the interaction to
    whichever component moves second:
      order_a: CATE first at the estimated value model, then the value model at
               the reference CATE.
      order_b: value model first at the estimated CATE, then CATE at the
               reference value model.
    shapley averages them (phi_tau, phi_v), so it does not depend on the order
    and phi_tau + phi_v = total.  Shares are normalised by the total and sum to 1;
    they are NOT bounded to [0, 1] (the paths can have opposite signs).  If the
    total is exactly 0 the shares are NaN and zero_denominator is True."""

    def regret(a, b):
        return W_ref - compute_welfare(target_topk(a * b, k), tau_true, ltv_true)

    W_ref = compute_welfare(target_topk(tau_ref * ltv_ref, k), tau_true, ltv_true)
    r_hh = regret(tau_hat, ltv_hat)
    r_rh = regret(tau_ref, ltv_hat)
    r_hr = regret(tau_hat, ltv_ref)
    r_rr = regret(tau_ref, ltv_ref)
    total = r_hh - r_rr
    a_tau, a_v = r_hh - r_rh, r_rh - r_rr
    b_v, b_tau = r_hh - r_hr, r_hr - r_rr
    phi_tau, phi_v = 0.5 * (a_tau + b_tau), 0.5 * (a_v + b_v)
    zero = bool(total == 0)

    def shares(t, v):
        if zero:
            return {"tau_share": float("nan"), "v_share": float("nan"), "tau": float(t), "v": float(v)}
        return {"tau_share": float(t / total), "v_share": float(v / total), "tau": float(t), "v": float(v)}

    return {
        "order_a": shares(a_tau, a_v),
        "order_b": shares(b_tau, b_v),
        "shapley": shares(phi_tau, phi_v),
        "phi_tau": float(phi_tau),
        "phi_v": float(phi_v),
        "total": float(total),
        "zero_denominator": zero,
    }
