"""Revision-41 additions: demonstrations requested by the fresh review panel.

Two analyses on the same 200 seeds:

  A. Diagnostic-loop demonstration. The regret decomposition claims to identify
     WHICH component's estimation error drives misranking. We compute the two
     telescoping terms of |tau*V - tauhat*Vhat| over the misranked set
     M = pi_star XOR pi_hat (oracle vs estimated top-k), under (i) the
     well-specified pipeline and (ii) CATE misspecification (login dropped).
     If the decomposition is diagnostic, the CATE-error term's share should
     rise sharply under CATE misspecification.

  B. End-to-end joint value-weighted model. The published S-Learner borrows the
     decomposed pipeline's value estimate. Here we run the fully end-to-end
     alternative the paper names but never ran: a single model trained to predict
     the value-weighted outcome (V_proxy * Y) directly, scored by the
     treated-minus-control difference. This is the un-handicapped joint comparator.

Usage:  python compute_v41_additions.py [N_RUNS]
"""

import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from simulation_v38 import (
    N_PLAYERS, N_RUNS as N_RUNS_DEFAULT, SEED_BASE,
)
from core.io import save_json
from core.dgp import (
    generate_main,
)
from core.estimators import (
    estimate_components, estimate_components_misspec,
)
from core.policy import (
    DEFAULT_BUDGET, target_topk, compute_welfare,
)

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else N_RUNS_DEFAULT
SEEDS = [SEED_BASE + r * 7 for r in range(N_RUNS)]


def diagnostic_terms(est_func, label):
    """Mean telescoping terms over the misranked set; share attributable to CATE."""
    cate_terms, value_terms, m_sizes = [], [], []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        data = generate_main(N_PLAYERS, rng)
        idx = np.arange(len(data['treatment']))
        tr, te = train_test_split(idx, test_size=0.30, random_state=seed)
        k = int(len(te) * DEFAULT_BUDGET)
        tau = data['true_tau'][te]
        V = data['true_ltv'][te]
        tauh, Vh, _ = est_func(data, tr, te)
        pi_star = set(target_topk(tau * V, k).tolist())
        pi_hat = set(target_topk(tauh * Vh, k).tolist())
        M = np.array(sorted(pi_star.symmetric_difference(pi_hat)), dtype=int)
        if len(M) == 0:
            continue
        # |tau*V - tauhat*Vhat| <= |tau-tauhat|*|V| + |tauhat|*|V-Vhat|
        t_cate = np.abs(tau[M] - tauh[M]) * np.abs(V[M])
        t_value = np.abs(tauh[M]) * np.abs(V[M] - Vh[M])
        cate_terms.append(float(t_cate.mean()))
        value_terms.append(float(t_value.mean()))
        m_sizes.append(len(M))
    a, b = float(np.mean(cate_terms)), float(np.mean(value_terms))
    return {
        'label': label,
        'cate_error_term': round(a, 0),
        'value_error_term': round(b, 0),
        'cate_share_pct': round(100 * a / (a + b), 1),
        'mean_misranked_set_size': round(float(np.mean(m_sizes)), 0),
    }


def end_to_end_joint():
    """Single model predicting the value-weighted outcome (V_proxy*Y) directly."""
    welfare = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        data = generate_main(N_PLAYERS, rng)
        idx = np.arange(len(data['treatment']))
        tr, te = train_test_split(idx, test_size=0.30, random_state=seed)
        k = int(len(te) * DEFAULT_BUDGET)
        X, W, Y = data['features'], data['treatment'], data['y_obs']
        Xtr, Xte, Wtr, Ytr = X[tr], X[te], W[tr], Y[tr]
        # Same misspecified value proxy form the decomposed pipeline trains on,
        # so the only difference from the borrowed-V S-Learner is joint vs. separate.
        rng_v = np.random.RandomState(hash(tuple(tr[:5].tolist())) % (2**31))
        vproxy = (12 * Xtr[:, 1] ** 0.9 + 35 * Xtr[:, 2]
                  + rng_v.normal(0, 200, len(Xtr))).clip(50, 8000)
        target = vproxy * Ytr  # value-weighted outcome
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=0)
        model.fit(np.column_stack([Xtr, Wtr]), target)
        n = len(Xte)
        f1 = model.predict(np.column_stack([Xte, np.ones(n)]))
        f0 = model.predict(np.column_stack([Xte, np.zeros(n)]))
        score = f1 - f0
        tau, V = data['true_tau'][te], data['true_ltv'][te]
        welfare.append(compute_welfare(target_topk(score, k), tau, V))
    return [round(float(np.mean(welfare)), 0), round(float(np.std(welfare)), 0)]


def main():
    print(f"Revision-41 additions | N_RUNS={N_RUNS}, kappa={DEFAULT_BUDGET}")
    out = {}

    print("A. Diagnostic loop (telescoping terms on misranked set) ...")
    out['diagnostic_wellspec'] = diagnostic_terms(estimate_components, 'well-specified')
    out['diagnostic_misspec'] = diagnostic_terms(estimate_components_misspec, 'CATE misspecified')
    for key in ['diagnostic_wellspec', 'diagnostic_misspec']:
        d = out[key]
        print(f"   [{d['label']:18}] CATE-term={d['cate_error_term']:.0f} "
              f"value-term={d['value_error_term']:.0f} CATE share={d['cate_share_pct']}% "
              f"|M|~{d['mean_misranked_set_size']:.0f}")

    print("B. End-to-end joint value-weighted model ...")
    out['end_to_end_joint'] = end_to_end_joint()
    print(f"   end-to-end joint welfare = {out['end_to_end_joint'][0]:.0f} +/- {out['end_to_end_joint'][1]:.0f}")
    print("   (compare: borrowed-V S-Learner 24,310; CATE x LTV 16,427)")

    out['_meta'] = {'n_runs': N_RUNS, 'kappa': DEFAULT_BUDGET, 'seeds_base': SEED_BASE}
    path = save_json(out, 'simulation_v41_additions.json')
    print(f"Saved {path}")


if __name__ == '__main__':
    main()
