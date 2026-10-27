"""Revision-40 additions: results requested in peer review.

Runs three analyses, all on the same 200 seeds as the published simulation so
the numbers are comparable with simulation_v29.json:

  A. Asymmetric-moderation DGP. Replaces the symmetric churn moderation
     m(p) = 4 p (1 - p) with an asymmetric form that attenuates faster at high
     churn than at low churn (Ascarza 2018). This is a structurally different
     second DGP that tests whether the "when each component earns its place"
     pattern, and the calibration-picks-identity behaviour, survive a change to
     the moderation structure.

  B. S-Learner in the misspecification and null DGPs. The published tables omit
     the S-Learner from these scenarios; we add it so the decomposition's
     "robustness under component misspecification" claim is adjudicated against
     the dominant alternative.

  C. Clip-adjusted (effective-tau) welfare for the main table. The headline
     welfare is scored on nominal tau; here we re-score on the realised
     (clip-adjusted) effective tau = y1_prob - y0_prob to check the bias is
     small and non-differential across strategies.

(The harm-gate simulation formerly here is now study_harm.py.)

Usage:  python compute_revision_additions.py [N_RUNS]
"""

import sys
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

from simulation_v38 import (
    N_PLAYERS, N_RUNS as N_RUNS_DEFAULT, SEED_BASE, run_single_simulation,
)
from core.io import save_json
from core.dgp import generate_main, generate_null, generate_asym
from core.estimators import (
    estimate_components, _estimate_ltv_churn, _estimate_slearner,
)
from core.policy import (
    calibrate_theta,
    DEFAULT_BUDGET, f_churn_vec, target_topk, compute_welfare,
)

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else N_RUNS_DEFAULT
SEEDS = [SEED_BASE + r * 7 for r in range(N_RUNS)]

def _mean_sd(x):
    return [round(float(np.mean(x)), 0), round(float(np.std(x)), 0)]


# ---------------------------------------------------------------------------
# A. Asymmetric-moderation DGP
# ---------------------------------------------------------------------------
def run_asymmetric():
    strategy_names = ['CATE Only', 'CATE x LTV', 'Full Framework (Est.)',
                      'Full Framework (Cal.)', 'Oracle hindsight (theta0)', 'S-Learner']
    welfare = {s: [] for s in strategy_names}
    cal_configs = []
    for seed in SEEDS:
        results, _, _, cal_cfg = run_single_simulation(
            seed, dgp_func=generate_asym)
        for s in strategy_names:
            welfare[s].append(results[s]['welfare'])
        cal_configs.append(cal_cfg)
    out = {s: _mean_sd(welfare[s]) for s in strategy_names}
    # how often did calibration pick the near-identity config?
    identity = {'pL': 0.00, 'pML': 0.01, 'pMH': 0.99, 'phi': 1.0}
    n_identity = sum(1 for c in cal_configs if c == identity)
    out['_cal_identity_count'] = [n_identity, N_RUNS]
    out['_cal_most_common'] = Counter(
        str(sorted(c.items())) for c in cal_configs).most_common(3)
    return out


# ---------------------------------------------------------------------------
# B. S-Learner in misspecification and null DGPs
# ---------------------------------------------------------------------------
def _slearner_welfare_for_dgp(dgp_func):
    """Lean S-Learner welfare loop (no calibration/frontier) for a given DGP."""
    welfare = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        data = dgp_func(N_PLAYERS, rng)
        idx = np.arange(len(data['treatment']))
        train_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=seed)
        k = int(len(test_idx) * DEFAULT_BUDGET)
        true_tau = data['true_tau'][test_idx]
        true_ltv = data['true_ltv'][test_idx]
        X_tr, X_te = data['features'][train_idx], data['features'][test_idx]
        W_tr, Y_tr = data['treatment'][train_idx], data['y_obs'][train_idx]
        ltv_hat, _ = _estimate_ltv_churn(X_tr, X_te, W_tr, Y_tr, seed)
        sl = _estimate_slearner(X_tr, X_te, W_tr, Y_tr, ltv_hat)
        welfare.append(compute_welfare(target_topk(sl, k), true_tau, true_ltv))
    return welfare


def run_slearner_in_scenarios():
    out = {}
    # Misspecification only changes the (T-Learner) CATE features; the S-Learner
    # is a joint model over all features and does not use the decomposed CATE, so
    # its welfare equals the main-DGP S-Learner exactly. We compute the main-DGP
    # S-Learner here and report the identity rather than re-running misspec.
    out['misspec_slearner'] = _mean_sd(_slearner_welfare_for_dgp(generate_main))
    out['misspec_note'] = 'identical to main-DGP S-Learner by construction (joint model, all features)'
    # Null DGP (no CATE-churn interaction): genuinely different data.
    out['null_slearner'] = _mean_sd(_slearner_welfare_for_dgp(generate_null))
    return out


# ---------------------------------------------------------------------------
# C. Clip-adjusted (effective-tau) welfare, main DGP, kappa=0.30
# ---------------------------------------------------------------------------
def run_effective_welfare():
    strategy_names = ['CATE Only', 'CATE x LTV', 'Full Framework (Cal.)',
                      'Oracle hindsight (theta0)', 'S-Learner']
    nominal = {s: [] for s in strategy_names}
    effective = {s: [] for s in strategy_names}
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        data = generate_main(N_PLAYERS, rng)
        n = len(data['treatment'])
        idx = np.arange(n)
        train_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=seed)
        k = int(len(test_idx) * DEFAULT_BUDGET)

        true_tau = data['true_tau'][test_idx]
        true_ltv = data['true_ltv'][test_idx]
        true_churn = data['true_churn'][test_idx]
        eff_tau = (data['y1_prob'][test_idx] - data['y0_prob'][test_idx])

        tau_hat, ltv_hat, churn_hat = estimate_components(data, train_idx, test_idx, seed=seed)
        g_oracle = f_churn_vec(true_churn)
        cal_cfg = calibrate_theta(data, train_idx, seed, DEFAULT_BUDGET)
        g_cal = f_churn_vec(churn_hat, **cal_cfg)

        X_tr, X_te = data['features'][train_idx], data['features'][test_idx]
        W_tr, Y_tr = data['treatment'][train_idx], data['y_obs'][train_idx]
        sl_scores = _estimate_slearner(X_tr, X_te, W_tr, Y_tr, ltv_hat)

        scores = {
            'CATE Only': tau_hat,
            'CATE x LTV': tau_hat * ltv_hat,
            'Full Framework (Cal.)': tau_hat * ltv_hat * g_cal,
            'Oracle hindsight (theta0)': true_tau * true_ltv * g_oracle,
            'S-Learner': sl_scores,
        }
        for s, sc in scores.items():
            tgt = target_topk(sc, k)
            nominal[s].append(float(np.sum(true_tau[tgt] * true_ltv[tgt])))
            effective[s].append(float(np.sum(eff_tau[tgt] * true_ltv[tgt])))
    out = {}
    for s in strategy_names:
        nm = float(np.mean(nominal[s]))
        ef = float(np.mean(effective[s]))
        out[s] = {
            'nominal': round(nm, 0),
            'effective': round(ef, 0),
            'pct_overstate': round(100 * (nm - ef) / ef, 1) if ef else None,
        }
    return out


def main():
    print(f"Revision-40 additions | N_RUNS={N_RUNS}, kappa={DEFAULT_BUDGET}")
    out = {}

    print("A. Asymmetric-moderation DGP ...")
    out['asymmetric_dgp'] = run_asymmetric()
    a = out['asymmetric_dgp']
    for s in ['CATE Only', 'CATE x LTV', 'Full Framework (Cal.)',
              'Oracle hindsight (theta0)', 'S-Learner']:
        print(f"   {s:28} W={a[s][0]:>8.0f} +/- {a[s][1]:.0f}")
    print(f"   calibration picked identity in {a['_cal_identity_count'][0]}/{a['_cal_identity_count'][1]} runs")

    print("B. S-Learner in misspec / null ...")
    out['slearner_scenarios'] = run_slearner_in_scenarios()
    b = out['slearner_scenarios']
    print(f"   misspec S-Learner W={b['misspec_slearner'][0]:.0f}; null S-Learner W={b['null_slearner'][0]:.0f}")

    print("C. Effective (clip-adjusted) welfare ...")
    out['effective_welfare'] = run_effective_welfare()
    for s, v in out['effective_welfare'].items():
        print(f"   {s:28} nominal={v['nominal']:>8.0f} effective={v['effective']:>8.0f} overstate={v['pct_overstate']}%")

    out['_meta'] = {'n_runs': N_RUNS, 'kappa': DEFAULT_BUDGET, 'seeds_base': SEED_BASE}
    path = save_json(out, 'simulation_v40_additions.json')
    print(f"Saved {path}")


if __name__ == '__main__':
    main()
