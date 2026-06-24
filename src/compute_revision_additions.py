"""Revision-40 additions: results requested in peer review.

Runs four analyses, all on the same 200 seeds as the published simulation so
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

  D. Harm-gate simulation. Adds a harm covariate correlated with LTV, shows the
     unguarded tau*V score concentrates bonuses on high-harm players, then
     applies a harm-protection weight h(x) and reports the welfare cost and the
     harm-exposure reduction (Corollary 1 made concrete).

Usage:  python compute_revision_additions.py [N_RUNS]
"""

import json
import os
import sys
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

from simulation_v38 import (
    N_PLAYERS, N_RUNS as N_RUNS_DEFAULT, DEFAULT_BUDGET, BUDGET_FRACTIONS,
    SEED_BASE, TREATMENT_PROB, POP_MEAN, POP_SD, _standardise,
    generate_population, generate_population_null,
    estimate_components, estimate_components_misspec, _estimate_ltv_churn,
    _estimate_slearner, f_churn_vec, target_topk, compute_welfare,
    compute_risk_exposure, calibrate_theta, run_single_simulation,
)

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else N_RUNS_DEFAULT
SEEDS = [SEED_BASE + r * 7 for r in range(N_RUNS)]

# Normalising constant for the asymmetric moderation p(1-p)^3 (max at p=0.25).
_ASYM_PEAK = 0.25 * (1 - 0.25) ** 3  # = 0.10546875


def churn_moderation_asym(p):
    """Asymmetric inverse-U: peaks near p=0.25, decays faster at high churn."""
    return np.clip(p * (1.0 - p) ** 3 / _ASYM_PEAK, 0.0, 1.0)


def generate_population_asym(n, rng):
    """Main DGP but with asymmetric churn moderation (Ascarza-style)."""
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
    churn_moderation = churn_moderation_asym(true_churn)  # asymmetric
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
        'treatment': treatment, 'y_obs': y_obs, 'y0': y0, 'y1': y1,
        'true_tau': true_tau, 'true_ltv': true_ltv, 'true_churn': true_churn,
        'y0_prob': y0_prob, 'y1_prob': y1_prob,
    }


def generate_population_harm(n, rng):
    """Main DGP plus a harm covariate correlated with LTV (via spend)."""
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

    # Harm score in [0,1]: rises with spend (the LTV driver), so harm and value
    # are positively correlated -- value-weighting will chase harmful players.
    harm_logit = 1.1 * z_spend + 0.4 * z_login - 0.3 + rng.normal(0, 0.5, n)
    true_harm = (1.0 / (1.0 + np.exp(-harm_logit))).clip(0.0, 1.0)

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
        'treatment': treatment, 'y_obs': y_obs, 'y0': y0, 'y1': y1,
        'true_tau': true_tau, 'true_ltv': true_ltv, 'true_churn': true_churn,
        'true_harm': true_harm, 'y0_prob': y0_prob, 'y1_prob': y1_prob,
    }


def h_protection(harm, h_lo=0.60, h_hi=0.90):
    """Harm-protection weight: 1 below h_lo, ramps to 0 above h_hi."""
    h = np.where(harm <= h_lo, 1.0,
                 np.where(harm >= h_hi, 0.0,
                          1.0 - (harm - h_lo) / (h_hi - h_lo)))
    return h


def _mean_sd(x):
    return [round(float(np.mean(x)), 0), round(float(np.std(x)), 0)]


# ---------------------------------------------------------------------------
# A. Asymmetric-moderation DGP
# ---------------------------------------------------------------------------
def run_asymmetric():
    strategy_names = ['CATE Only', 'CATE x LTV', 'Full Framework (Est.)',
                      'Full Framework (Cal.)', 'Full Framework (Oracle)', 'S-Learner']
    welfare = {s: [] for s in strategy_names}
    cal_configs = []
    for seed in SEEDS:
        results, _, _, cal_cfg = run_single_simulation(
            seed, dgp_func=generate_population_asym)
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
        ltv_hat, _ = _estimate_ltv_churn(X_tr, X_te, W_tr, Y_tr, train_idx)
        sl = _estimate_slearner(X_tr, X_te, W_tr, Y_tr, ltv_hat)
        welfare.append(compute_welfare(target_topk(sl, k), true_tau, true_ltv))
    return welfare


def run_slearner_in_scenarios():
    out = {}
    # Misspecification only changes the (T-Learner) CATE features; the S-Learner
    # is a joint model over all features and does not use the decomposed CATE, so
    # its welfare equals the main-DGP S-Learner exactly. We compute the main-DGP
    # S-Learner here and report the identity rather than re-running misspec.
    out['misspec_slearner'] = _mean_sd(_slearner_welfare_for_dgp(generate_population))
    out['misspec_note'] = 'identical to main-DGP S-Learner by construction (joint model, all features)'
    # Null DGP (no CATE-churn interaction): genuinely different data.
    out['null_slearner'] = _mean_sd(_slearner_welfare_for_dgp(generate_population_null))
    return out


# ---------------------------------------------------------------------------
# C. Clip-adjusted (effective-tau) welfare, main DGP, kappa=0.30
# ---------------------------------------------------------------------------
def run_effective_welfare():
    strategy_names = ['CATE Only', 'CATE x LTV', 'Full Framework (Cal.)',
                      'Full Framework (Oracle)', 'S-Learner']
    nominal = {s: [] for s in strategy_names}
    effective = {s: [] for s in strategy_names}
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        data = generate_population(N_PLAYERS, rng)
        n = len(data['treatment'])
        idx = np.arange(n)
        train_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=seed)
        k = int(len(test_idx) * DEFAULT_BUDGET)

        true_tau = data['true_tau'][test_idx]
        true_ltv = data['true_ltv'][test_idx]
        true_churn = data['true_churn'][test_idx]
        eff_tau = (data['y1_prob'][test_idx] - data['y0_prob'][test_idx])

        tau_hat, ltv_hat, churn_hat = estimate_components(data, train_idx, test_idx)
        g_oracle = f_churn_vec(true_churn)
        cal_cfg = calibrate_theta(data, train_idx, seed, k)
        g_cal = f_churn_vec(churn_hat, **cal_cfg)

        X_tr, X_te = data['features'][train_idx], data['features'][test_idx]
        W_tr, Y_tr = data['treatment'][train_idx], data['y_obs'][train_idx]
        sl_scores = _estimate_slearner(X_tr, X_te, W_tr, Y_tr, ltv_hat)

        scores = {
            'CATE Only': tau_hat,
            'CATE x LTV': tau_hat * ltv_hat,
            'Full Framework (Cal.)': tau_hat * ltv_hat * g_cal,
            'Full Framework (Oracle)': true_tau * true_ltv * g_oracle,
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


# ---------------------------------------------------------------------------
# D. Harm-gate simulation
# ---------------------------------------------------------------------------
def _harm_block(welfare_ung, harm_ung, welfare_grd, harm_grd):
    uw, gw = float(np.mean(welfare_ung)), float(np.mean(welfare_grd))
    uh, gh = float(np.mean(harm_ung)), float(np.mean(harm_grd))
    return {
        'unguarded': {'welfare': round(uw, 0), 'harm_exposure': round(uh, 3)},
        'guarded': {'welfare': round(gw, 0), 'harm_exposure': round(gh, 3)},
        'welfare_cost_abs': round(uw - gw, 0),
        'welfare_cost_pct': round(100 * (uw - gw) / uw, 1) if uw else None,
        'harm_reduction_abs': round(uh - gh, 3),
        'harm_reduction_pct': round(100 * (uh - gh) / uh, 1) if uh else None,
    }


def run_harm_gate():
    # Oracle rule (true components) -- clean demonstration of Corollary 1.
    o_uw, o_uh, o_gw, o_gh = [], [], [], []
    # Estimated rule -- the deployment regime (can differ in sign of the cost).
    e_uw, e_uh, e_gw, e_gh = [], [], [], []
    pop_harm = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        data = generate_population_harm(N_PLAYERS, rng)
        idx = np.arange(len(data['treatment']))
        train_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=seed)
        k = int(len(test_idx) * DEFAULT_BUDGET)

        true_tau = data['true_tau'][test_idx]
        true_ltv = data['true_ltv'][test_idx]
        true_harm = data['true_harm'][test_idx]
        h = h_protection(true_harm)
        pop_harm.append(float(np.mean(true_harm)))

        # --- Oracle rule ---
        t = target_topk(true_tau * true_ltv, k)
        o_uw.append(compute_welfare(t, true_tau, true_ltv)); o_uh.append(float(np.mean(true_harm[t])))
        t = target_topk(true_tau * true_ltv * h, k)
        o_gw.append(compute_welfare(t, true_tau, true_ltv)); o_gh.append(float(np.mean(true_harm[t])))

        # --- Estimated rule ---
        tau_hat, ltv_hat, _ = estimate_components(data, train_idx, test_idx)
        t = target_topk(tau_hat * ltv_hat, k)
        e_uw.append(compute_welfare(t, true_tau, true_ltv)); e_uh.append(float(np.mean(true_harm[t])))
        t = target_topk(tau_hat * ltv_hat * h, k)
        e_gw.append(compute_welfare(t, true_tau, true_ltv)); e_gh.append(float(np.mean(true_harm[t])))

    return {
        'population_mean_harm': round(float(np.mean(pop_harm)), 3),
        'oracle': _harm_block(o_uw, o_uh, o_gw, o_gh),
        'estimated': _harm_block(e_uw, e_uh, e_gw, e_gh),
    }


def main():
    print(f"Revision-40 additions | N_RUNS={N_RUNS}, kappa={DEFAULT_BUDGET}")
    out = {}

    print("A. Asymmetric-moderation DGP ...")
    out['asymmetric_dgp'] = run_asymmetric()
    a = out['asymmetric_dgp']
    for s in ['CATE Only', 'CATE x LTV', 'Full Framework (Cal.)',
              'Full Framework (Oracle)', 'S-Learner']:
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

    print("D. Harm-gate simulation ...")
    out['harm_gate'] = run_harm_gate()
    h = out['harm_gate']
    print(f"   pop mean harm={h['population_mean_harm']}")
    for reg in ['oracle', 'estimated']:
        b = h[reg]
        print(f"   [{reg}] unguarded harm={b['unguarded']['harm_exposure']} (W={b['unguarded']['welfare']:.0f}) "
              f"-> guarded harm={b['guarded']['harm_exposure']} (W={b['guarded']['welfare']:.0f}); "
              f"welfare cost={b['welfare_cost_pct']}%, harm reduction={b['harm_reduction_pct']}%")

    out['_meta'] = {'n_runs': N_RUNS, 'kappa': DEFAULT_BUDGET, 'seeds_base': SEED_BASE}
    path = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                        'simulation_v40_additions.json')
    with open(path, 'w') as fp:
        json.dump(out, fp, indent=2)
    print(f"Saved {path}")


if __name__ == '__main__':
    main()
