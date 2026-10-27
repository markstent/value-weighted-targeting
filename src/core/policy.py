"""Targeting policy primitives: risk modulation g, top-k selection, scoring."""

import numpy as np

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
