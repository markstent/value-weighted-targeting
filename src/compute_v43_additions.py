"""Revision-43 additions: robustness runs from the third review.

  A. corr(tau, V) sweep. The main DGP makes value roughly independent of the
     treatment effect. We inject a controlled dependence V <- V + gamma*c*z(tau_base)
     and sweep gamma over negative / zero / positive, to test whether value
     weighting's advantage is an artefact of independent (rather than competing)
     value signal.

  B. Heavy-tailed LTV. Real gaming value is extreme-tailed. We replace the additive
     LTV with a multiplicative log-normal tail (orders-of-magnitude spread) and
     re-run the main comparison.

  C. Observational (plug-in) diagnostic. The oracle regret decomposition needs true
     tau, V. We give the deployable analogue: replace the oracle with a richer
     cross-fitted reference estimator (X-Learner tau, all-feature V) and compute the
     telescoping terms from the disagreement between the reference and the deployed
     (T-Learner, misspecified-V) estimates. If this observable proxy still flags the
     degraded component, the diagnostic is usable without oracle quantities.

Usage:  python compute_v43_additions.py [N_RUNS]
"""

import sys
import numpy as np
from sklearn.model_selection import train_test_split

from simulation_v38 import (
    N_PLAYERS, N_RUNS as N_RUNS_DEFAULT, SEED_BASE,
)
from core.io import save_json
from core.dgp import (
    generate_main, make_corr_dgp, generate_heavytail,
)
from core.estimators import (
    VALUE_STREAM_V43, seed_from,
    estimate_components, estimate_components_misspec, _estimate_slearner, _estimate_cate_xlearner,
)
from core.policy import (
    DEFAULT_BUDGET, target_topk, compute_welfare,
)

from sklearn.ensemble import GradientBoostingRegressor

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else N_RUNS_DEFAULT
SEEDS = [SEED_BASE + r * 7 for r in range(N_RUNS)]


def _strategy_welfare(dgp, kappa=DEFAULT_BUDGET):
    """Lean: CATE Only, CATE x LTV, S-Learner welfare (no calibration/frontier)."""
    co, cl, sl = [], [], []
    corr_tv = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        data = dgp(N_PLAYERS, rng)
        idx = np.arange(len(data['treatment']))
        tr, te = train_test_split(idx, test_size=0.30, random_state=seed)
        k = int(len(te) * kappa)
        tau, V = data['true_tau'][te], data['true_ltv'][te]
        corr_tv.append(float(np.corrcoef(data['true_tau'], data['true_ltv'])[0, 1]))
        tauh, vh, _ = estimate_components(data, tr, te, seed=seed)
        Xtr, Xte = data['features'][tr], data['features'][te]
        Wtr, Ytr = data['treatment'][tr], data['y_obs'][tr]
        sl_score = _estimate_slearner(Xtr, Xte, Wtr, Ytr, vh)
        co.append(compute_welfare(target_topk(tauh, k), tau, V))
        cl.append(compute_welfare(target_topk(tauh * vh, k), tau, V))
        sl.append(compute_welfare(target_topk(sl_score, k), tau, V))
    return {
        'corr_tau_V': round(float(np.mean(corr_tv)), 2),
        'CATE Only': round(float(np.mean(co)), 0),
        'CATE x LTV': round(float(np.mean(cl)), 0),
        'S-Learner': round(float(np.mean(sl)), 0),
        'LTV_x_advantage_pct': round(100 * (np.mean(cl) - np.mean(co)) / np.mean(co), 1),
    }


def observational_diagnostic(est_func):
    """Plug-in telescoping terms using a cross-fitted reference instead of the oracle."""
    cate_terms, value_terms = [], []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        data = generate_main(N_PLAYERS, rng)
        idx = np.arange(len(data['treatment']))
        tr, te = train_test_split(idx, test_size=0.30, random_state=seed)
        k = int(len(te) * DEFAULT_BUDGET)
        tauh, vh, _ = est_func(data, tr, te, seed=seed)               # deployed (T-Learner, misspec V)
        # Reference: richer X-Learner tau on ALL features, all-feature value model.
        Xtr, Xte = data['features'][tr], data['features'][te]
        Wtr, Ytr = data['treatment'][tr], data['y_obs'][tr]
        tau_ref = _estimate_cate_xlearner(Xtr, Xte, Wtr, Ytr)
        # Richer value reference: deeper/larger ensemble on the same observable proxy
        # target, so disagreement with the deployed value model is observable (no oracle).
        rng_v = np.random.RandomState(seed_from(seed, VALUE_STREAM_V43))
        vproxy = (12 * Xtr[:, 1] ** 0.9 + 35 * Xtr[:, 2]
                  + rng_v.normal(0, 200, len(Xtr)))
        vref_model = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05, random_state=1)
        vref_model.fit(Xtr, vproxy)
        vh_ref = vref_model.predict(Xte).clip(50, 8000)
        pi_dep = set(target_topk(tauh * vh, k).tolist())
        pi_ref = set(target_topk(tau_ref * vh_ref, k).tolist())
        M = np.array(sorted(pi_dep.symmetric_difference(pi_ref)), dtype=int)
        if len(M) == 0:
            continue
        t_cate = np.abs(tau_ref[M] - tauh[M]) * np.abs(vh[M])
        t_value = np.abs(tauh[M]) * np.abs(vh_ref[M] - vh[M])
        cate_terms.append(float(t_cate.mean()))
        value_terms.append(float(t_value.mean()))
    a, b = float(np.mean(cate_terms)), float(np.mean(value_terms))
    return {'cate_term': round(a, 0), 'value_term': round(b, 0),
            'cate_share_pct': round(100 * a / (a + b), 1)}


def main():
    print(f"Revision-43 additions | N_RUNS={N_RUNS}, kappa={DEFAULT_BUDGET}")
    out = {}

    print("A. corr(tau, V) sweep ...")
    out['corr_sweep'] = {}
    for label, gamma in [('negative', -1.0), ('zero', 0.0), ('positive', 1.0)]:
        out['corr_sweep'][label] = _strategy_welfare(make_corr_dgp(gamma))
        r = out['corr_sweep'][label]
        print(f"   gamma={label:8} corr={r['corr_tau_V']:+.2f}  CATE={r['CATE Only']:.0f} "
              f"CATExLTV={r['CATE x LTV']:.0f} ({r['LTV_x_advantage_pct']:+.1f}%) S-L={r['S-Learner']:.0f}")

    print("B. Heavy-tailed LTV ...")
    out['heavy_tail'] = _strategy_welfare(generate_heavytail)
    h = out['heavy_tail']
    print(f"   corr={h['corr_tau_V']:+.2f} CATE={h['CATE Only']:.0f} "
          f"CATExLTV={h['CATE x LTV']:.0f} ({h['LTV_x_advantage_pct']:+.1f}%) S-L={h['S-Learner']:.0f}")

    print("C. Observational (plug-in) diagnostic ...")
    out['obs_diagnostic_wellspec'] = observational_diagnostic(estimate_components)
    out['obs_diagnostic_misspec'] = observational_diagnostic(estimate_components_misspec)
    for key, lab in [('obs_diagnostic_wellspec', 'well-specified'),
                     ('obs_diagnostic_misspec', 'CATE misspecified')]:
        d = out[key]
        print(f"   [{lab:18}] CATE-term={d['cate_term']:.0f} value-term={d['value_term']:.0f} "
              f"CATE share={d['cate_share_pct']}%")

    out['_meta'] = {'n_runs': N_RUNS, 'kappa': DEFAULT_BUDGET, 'seeds_base': SEED_BASE}
    path = save_json(out, 'simulation_v43_additions.json')
    print(f"Saved {path}")


if __name__ == '__main__':
    main()
