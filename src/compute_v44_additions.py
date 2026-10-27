"""Revision-44 additions: results requested in the fourth-review pass.

Runs three analyses, all on the same 200 seeds as the published simulation
(SEED_BASE + 7r) so the numbers are comparable with simulation_v29.json:

  A. Budget-sensitivity INFERENCE + crossover budget kappa* (review #7). The
     published budget table (Table: budget sensitivity) reports welfare means
     only, while the paper's title claim -- that pure CATE beats value-weighting
     at tight budgets and the ranking reverses as the budget grows -- rests on
     it. We add paired t-tests, Holm-Bonferroni correction across the budget
     family, Cohen's d, and 95% CIs for the CATE Only vs CATE x LTV comparison,
     and estimate the crossover budget kappa* (where CATE x LTV overtakes CATE
     Only) with a bootstrap CI, for both the symmetric main DGP and the
     asymmetric-moderation DGP.

  B. AIPW (doubly robust) calibration (review #6). The plug-in calibration
     objective, sum of tau_hat*ltv_hat over the selected set, is maximised by
     g=1 for any estimates (top-k tautology), so it is not evidence about
     theta. AIPW is the primary objective: on the validation fold the
     doubly-robust score Gamma (known propensity e=0.5, outcome models mu0,
     mu1 cross-fit within the validation fold) is weighted by ltv_hat and
     summed over the selected set. We report the identity-selection rate and
     selected configs of both objectives, and the true-welfare regret of the
     calibrated rule relative to the identity rule.

  C. Adversarial CATE misspecification (review #8). The published
     misspecification drops login frequency from the CATE model; because login
     enters LTV linearly, the value model recovers the dropped signal, which is
     favourable to value weighting by construction. Here we drop a feature that
     drives tau but is ABSENT from V (bonus_history), so the value model cannot
     recover it. This bounds the claim that value weighting helps most when the
     treatment-effect model is weakest.

Usage:  python compute_v44_additions.py [N_RUNS]
"""

import sys
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

from simulation_v38 import (
    N_PLAYERS, N_RUNS as N_RUNS_DEFAULT, SEED_BASE,
)
from core.io import save_json
from core.dgp import (
    generate_main, generate_adv, generate_asym, TREATMENT_PROB,
)
from core.estimators import (
    LTV_STREAM, estimate_components, _estimate_ltv_churn, _estimate_cate_tlearner, _estimate_slearner,
)
from core.policy import (
    DEFAULT_BUDGET, f_churn_vec, target_topk, compute_welfare, THETA_GRID, DEFAULT_THETA_IDX, calibrate_theta_all,
)
from core.stats import (
    holm_bonferroni, paired_test,
)

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else N_RUNS_DEFAULT
SEEDS = [SEED_BASE + r * 7 for r in range(N_RUNS)]


# ---------------------------------------------------------------------------
# A. Budget-sensitivity inference + crossover kappa*
# ---------------------------------------------------------------------------

# Fine grid for the crossover search; the four reported budgets are a subset.
KAPPA_GRID = [round(0.05 + 0.025 * i, 3) for i in range(19)]  # 0.05 .. 0.50
REPORTED_BUDGETS = [0.10, 0.20, 0.30, 0.50]


def _fit_test_components(seed, dgp_func):
    """Fit T-Learner + LTV once per seed; return test-set arrays.

    Components do not depend on the budget kappa, so we fit once and sweep
    kappa by re-selecting the top-k set. tau_hat and ltv_hat are the estimated
    components; true_tau/true_ltv are the oracle test-set quantities used to
    score welfare.
    """
    rng = np.random.RandomState(seed)
    data = dgp_func(N_PLAYERS, rng)
    n = len(data['treatment'])
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(indices, test_size=0.30,
                                           random_state=seed)
    tau_hat, ltv_hat, _ = estimate_components(data, train_idx, test_idx, seed=seed)
    true_tau = data['true_tau'][test_idx]
    true_ltv = data['true_ltv'][test_idx]
    return tau_hat, ltv_hat, true_tau, true_ltv


def _welfare_at_kappa(scores, kappa, true_tau, true_ltv):
    k = int(len(scores) * kappa)
    targeted = target_topk(scores, k)
    return compute_welfare(targeted, true_tau, true_ltv)


def _budget_curves(dgp_func):
    """Per-seed welfare for CATE Only and CATE x LTV across the kappa grid."""
    cate = {kappa: [] for kappa in KAPPA_GRID}
    ltv = {kappa: [] for kappa in KAPPA_GRID}
    for i, seed in enumerate(SEEDS):
        if (i + 1) % 50 == 0:
            print(f"    seed {i + 1}/{N_RUNS}")
        tau_hat, ltv_hat, true_tau, true_ltv = _fit_test_components(seed, dgp_func)
        s_cate = tau_hat
        s_ltv = tau_hat * ltv_hat
        for kappa in KAPPA_GRID:
            cate[kappa].append(_welfare_at_kappa(s_cate, kappa, true_tau, true_ltv))
            ltv[kappa].append(_welfare_at_kappa(s_ltv, kappa, true_tau, true_ltv))
    return cate, ltv


def _crossover_kappa(cate, ltv, kappa_grid):
    """Interpolated kappa where mean(CATE x LTV - CATE Only) crosses zero."""
    diffs = np.array([np.mean(np.array(ltv[k]) - np.array(cate[k]))
                      for k in kappa_grid])
    ks = np.array(kappa_grid)
    # find first sign change from negative to positive
    for j in range(len(ks) - 1):
        if diffs[j] <= 0 <= diffs[j + 1]:
            # linear interpolation of the zero crossing
            frac = -diffs[j] / (diffs[j + 1] - diffs[j])
            return float(ks[j] + frac * (ks[j + 1] - ks[j]))
    if diffs[0] > 0:
        return float(ks[0])   # already positive at the tightest budget
    return float('nan')       # no crossover in range (CATE Only wins throughout)


def _bootstrap_crossover(cate, ltv, kappa_grid, n_boot=2000, seed=0):
    """Bootstrap CI for the crossover kappa* by resampling replications."""
    rng = np.random.RandomState(seed)
    n = N_RUNS
    cate_arr = {k: np.array(cate[k]) for k in kappa_grid}
    ltv_arr = {k: np.array(ltv[k]) for k in kappa_grid}
    ks = np.array(kappa_grid)
    crossings = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs = np.array([np.mean(ltv_arr[k][idx] - cate_arr[k][idx])
                          for k in kappa_grid])
        star = float('nan')
        for j in range(len(ks) - 1):
            if diffs[j] <= 0 <= diffs[j + 1]:
                frac = -diffs[j] / (diffs[j + 1] - diffs[j])
                star = float(ks[j] + frac * (ks[j + 1] - ks[j]))
                break
        else:
            if diffs[0] > 0:
                star = float(ks[0])
        crossings.append(star)
    crossings = np.array([c for c in crossings if not np.isnan(c)])
    if len(crossings) == 0:
        return None
    return {
        'median': round(float(np.median(crossings)), 4),
        'ci_lo': round(float(np.percentile(crossings, 2.5)), 4),
        'ci_hi': round(float(np.percentile(crossings, 97.5)), 4),
        'frac_with_crossover': round(len(crossings) / n_boot, 3),
    }


def run_budget_inference(dgp_func, label):
    print(f"  [A] budget inference + crossover ({label})")
    cate, ltv = _budget_curves(dgp_func)

    # Paired inference at the four reported budgets (CATE x LTV vs CATE Only),
    # Holm-corrected across the budget family.
    per_budget = {}
    raw_p = []
    keys = []
    for kappa in REPORTED_BUDGETS:
        w_ltv = ltv[kappa]
        w_cate = cate[kappa]
        pt = paired_test(w_ltv, w_cate, f'{kappa} ltv vs cate')
        per_budget[str(kappa)] = {
            'cate_only_welfare': round(float(np.mean(w_cate)), 0),
            'cate_x_ltv_welfare': round(float(np.mean(w_ltv)), 0),
            'diff_ltv_minus_cate': pt['mean_diff'],
            'se_diff': pt['se_diff'],
            't_stat': pt['t_stat'],
            'p_value': pt['p_value'],
            'cohens_d': pt['cohens_d'],
            'ci_95': pt['ci_95'],
        }
        raw_p.append(pt['p_value'])
        keys.append(str(kappa))
    corrected = holm_bonferroni(raw_p)
    for i, key in enumerate(keys):
        per_budget[key]['p_holm'] = round(float(corrected[i]), 4)

    kappa_star = _crossover_kappa(cate, ltv, KAPPA_GRID)
    boot = _bootstrap_crossover(cate, ltv, KAPPA_GRID)
    return {
        'per_budget': per_budget,
        'kappa_star_point': round(kappa_star, 4) if not np.isnan(kappa_star) else None,
        'kappa_star_bootstrap': boot,
        'kappa_grid': KAPPA_GRID,
        'mean_diff_curve': [round(float(np.mean(np.array(ltv[k]) - np.array(cate[k]))), 1)
                            for k in KAPPA_GRID],
    }


# ---------------------------------------------------------------------------
# B. AIPW (doubly robust) calibration
# ---------------------------------------------------------------------------

def run_aipw_calibration():
    print("  [B] AIPW vs plug-in calibration selection")
    aipw_idx = []
    plugin_idx = []
    # true test welfare of the calibrated rule minus the identity rule.
    # The plug-in objective selects identity by construction (regret 0).
    # AIPW selected non-identity configs in 0 of 10 replications (identity
    # rate 0/10 on both DGPs); at
    # N=10000, kappa=0.30, 10 reps (seeds 42+7r) its mean regret is negative
    # on both the null DGP (-4400, SE 2563) and the main DGP (-1040, SE 684),
    # with large per-rep spread (regret of both signs). That is consistent
    # with AIPW selection being high-variance and, on average, not welfare
    # improving; it is not significantly different from 0 at 10 reps, so we
    # claim no welfare gain from calibration rather than a proven loss.
    # Motivation only: Proposition 3 implies no g improves on identity at the
    # population optimum (true tau and V); with estimated components this need
    # not hold, and here it is not observed.
    aipw_cost = []
    plugin_cost = []
    for i, seed in enumerate(SEEDS):
        if (i + 1) % 50 == 0:
            print(f"    seed {i + 1}/{N_RUNS}")
        rng = np.random.RandomState(seed)
        data = generate_main(N_PLAYERS, rng)
        n = len(data['treatment'])
        train_idx, test_idx = train_test_split(np.arange(n), test_size=0.30,
                                               random_state=seed)

        # deployed components on the full train -> test
        tau_hat, ltv_hat, churn_hat = estimate_components(
            data, train_idx, test_idx, seed=seed)
        true_tau = data['true_tau'][test_idx]
        true_ltv = data['true_ltv'][test_idx]
        w_identity = _welfare_at_kappa(tau_hat * ltv_hat, DEFAULT_BUDGET,
                                       true_tau, true_ltv)

        cals = calibrate_theta_all(data, train_idx, seed, DEFAULT_BUDGET)
        cfg_a = cals['aipw']
        a_idx = THETA_GRID.index(cfg_a)
        aipw_idx.append(a_idx)
        g_a = f_churn_vec(churn_hat, **cfg_a)
        w_a = _welfare_at_kappa(tau_hat * ltv_hat * g_a, DEFAULT_BUDGET,
                                true_tau, true_ltv)
        aipw_cost.append(w_a - w_identity)

        # plug-in selection on the same seed, for a matched comparison
        cfg_p = cals['plugin']
        p_idx = THETA_GRID.index(cfg_p) if cfg_p in THETA_GRID else DEFAULT_THETA_IDX
        plugin_idx.append(p_idx)
        g_p = f_churn_vec(churn_hat, **cfg_p)
        w_p = _welfare_at_kappa(tau_hat * ltv_hat * g_p, DEFAULT_BUDGET,
                                true_tau, true_ltv)
        plugin_cost.append(w_p - w_identity)

    def _summ(idxs, cost):
        c = Counter(idxs)
        cost = np.array(cost)
        return {
            'identity_selected': int(c.get(0, 0)),
            'n_runs': len(idxs),
            'identity_rate': round(c.get(0, 0) / len(idxs), 3),
            'selected_config_counts': {str(k): v for k, v in c.most_common()},
            'mean_true_welfare_vs_identity': round(float(cost.mean()), 1),
            'true_welfare_vs_identity_ci95': [
                round(float(cost.mean() - 1.96 * cost.std() / np.sqrt(len(cost))), 1),
                round(float(cost.mean() + 1.96 * cost.std() / np.sqrt(len(cost))), 1)],
        }

    return {
        'aipw': _summ(aipw_idx, aipw_cost),
        'plugin': _summ(plugin_idx, plugin_cost),
        'note': ('config index 0 is the near-identity g~1; index 3 is default '
                 'theta0. mean_true_welfare_vs_identity is the oracle test-welfare '
                 'of the calibrated rule minus the g=1 (CATE x LTV) rule. The plug-in '
                 'objective selects identity by construction (regret 0). '
                 'Proposition 3 implies no g improves on identity at the '
                 'population optimum (true tau and V); with estimated '
                 'components this need not hold, so a negative or zero mean '
                 'regret here is an empirical observation, not a consequence '
                 'of the proposition.'),
    }


# ---------------------------------------------------------------------------
# C. Adversarial CATE misspecification (dropped feature absent from V)
# ---------------------------------------------------------------------------

def estimate_components_misspec_adv(data, train_idx, test_idx, cate_method='tlearner', *,
                                    seed, stream=LTV_STREAM):
    """CATE model drops bonus_history (index 3); LTV and churn keep all features."""
    X = data['features']
    W = data['treatment']
    Y = data['y_obs']
    X_train, X_test = X[train_idx], X[test_idx]
    W_train, Y_train = W[train_idx], Y[train_idx]

    cate_cols = [0, 1, 2]  # drop bonus_history (index 3)
    tau_hat = _estimate_cate_tlearner(
        X_train[:, cate_cols], X_test[:, cate_cols], W_train, Y_train)
    ltv_hat, churn_hat = _estimate_ltv_churn(
        X_train, X_test, W_train, Y_train, seed, stream)
    return tau_hat, ltv_hat, churn_hat


def _run_adv_scenario(est_func, dgp_func, kappa):
    cate, ltv, sl = [], [], []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        data = dgp_func(N_PLAYERS, rng)
        n = len(data['treatment'])
        train_idx, test_idx = train_test_split(np.arange(n), test_size=0.30,
                                               random_state=seed)
        tau_hat, ltv_hat, _ = est_func(data, train_idx, test_idx, seed=seed)
        true_tau = data['true_tau'][test_idx]
        true_ltv = data['true_ltv'][test_idx]
        cate.append(_welfare_at_kappa(tau_hat, kappa, true_tau, true_ltv))
        ltv.append(_welfare_at_kappa(tau_hat * ltv_hat, kappa, true_tau, true_ltv))
        # S-Learner uses all features (sees bonus_history)
        sl_scores = _estimate_slearner(
            data['features'][train_idx], data['features'][test_idx],
            data['treatment'][train_idx], data['y_obs'][train_idx], ltv_hat)
        sl.append(_welfare_at_kappa(sl_scores, kappa, true_tau, true_ltv))
    return np.array(cate), np.array(ltv), np.array(sl)


def run_adversarial_misspec(kappa=0.30):
    print("  [C] adversarial misspecification (drop bonus_history, absent from V)")
    # adversarial DGP, adversarial dropout
    cate, ltv, sl = _run_adv_scenario(
        estimate_components_misspec_adv, generate_adv, kappa)
    # same adversarial DGP, well-specified estimation (reference)
    cate_ws, ltv_ws, sl_ws = _run_adv_scenario(
        estimate_components, generate_adv, kappa)

    pt = paired_test(list(ltv), list(cate), 'adv ltv vs cate')
    return {
        'kappa': kappa,
        'adversarial_dropout': {
            'cate_only': round(float(cate.mean()), 0),
            'cate_only_sd': round(float(cate.std()), 0),
            'cate_x_ltv': round(float(ltv.mean()), 0),
            'cate_x_ltv_sd': round(float(ltv.std()), 0),
            's_learner': round(float(sl.mean()), 0),
            'ltv_minus_cate': pt['mean_diff'],
            'ltv_minus_cate_pct': round(100 * pt['mean_diff'] / cate.mean(), 2),
            'p_value': pt['p_value'],
            'cohens_d': pt['cohens_d'],
            'ci_95': pt['ci_95'],
        },
        'well_specified_reference': {
            'cate_only': round(float(cate_ws.mean()), 0),
            'cate_x_ltv': round(float(ltv_ws.mean()), 0),
            's_learner': round(float(sl_ws.mean()), 0),
        },
        'interpretation': (
            'If ltv_minus_cate is <= 0 (value weighting does NOT beat pure CATE), '
            'the value model failed to rescue the dropped signal, bounding the '
            'favourable login-drop result in the published misspecification table.'),
    }


# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(f"v44 additions  (N_RUNS={N_RUNS})")
    print("=" * 70)

    out = {}
    out['budget_inference_main'] = run_budget_inference(generate_main, 'symmetric main')
    out['budget_inference_asym'] = run_budget_inference(generate_asym, 'asymmetric')
    out['aipw_calibration'] = run_aipw_calibration()
    out['adversarial_misspec'] = run_adversarial_misspec(kappa=0.30)
    out['adversarial_misspec_tight'] = run_adversarial_misspec(kappa=0.10)
    out['meta'] = {
        'n_runs': N_RUNS,
        'seeds': 'SEED_BASE + 7r',
        'kappa_grid': KAPPA_GRID,
        'reported_budgets': REPORTED_BUDGETS,
    }

    # ---- print summary ----
    print("\n" + "=" * 70)
    print("A. BUDGET INFERENCE (CATE x LTV minus CATE Only)")
    for label in ['budget_inference_main', 'budget_inference_asym']:
        print(f"\n  {label}:")
        for kappa, row in out[label]['per_budget'].items():
            print(f"    kappa={kappa}: diff={row['diff_ltv_minus_cate']:+,.0f} "
                  f"(d={row['cohens_d']}, p_holm={row['p_holm']}) "
                  f"CATE={row['cate_only_welfare']:,.0f} LTV={row['cate_x_ltv_welfare']:,.0f}")
        ks = out[label]['kappa_star_bootstrap']
        print(f"    kappa* point={out[label]['kappa_star_point']}, "
              f"bootstrap={ks}")

    print("\nB. AIPW vs PLUG-IN CALIBRATION")
    a = out['aipw_calibration']
    print(f"  AIPW  identity {a['aipw']['identity_selected']}/{a['aipw']['n_runs']} "
          f"({a['aipw']['identity_rate']}); configs={a['aipw']['selected_config_counts']}")
    print(f"  Plug  identity {a['plugin']['identity_selected']}/{a['plugin']['n_runs']} "
          f"({a['plugin']['identity_rate']}); configs={a['plugin']['selected_config_counts']}")

    print("\nC. ADVERSARIAL MISSPECIFICATION")
    for key in ['adversarial_misspec', 'adversarial_misspec_tight']:
        c = out[key]['adversarial_dropout']
        r = out[key]['well_specified_reference']
        print(f"  kappa={out[key]['kappa']}: [adversarial dropout] "
              f"CATE={c['cate_only']:,.0f} LTV={c['cate_x_ltv']:,.0f} "
              f"(diff {c['ltv_minus_cate']:+,.0f} = {c['ltv_minus_cate_pct']:+.1f}%, "
              f"p={c['p_value']}) SL={c['s_learner']:,.0f}")
        print(f"           [well-specified ref] "
              f"CATE={r['cate_only']:,.0f} LTV={r['cate_x_ltv']:,.0f} SL={r['s_learner']:,.0f}")

    output_path = save_json(out, 'simulation_v44_additions.json')
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
