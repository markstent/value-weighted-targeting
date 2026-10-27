"""Hindsight and feasible tau*V oracle welfare (g=1) for the main scenario, kappa=0.30.

Hindsight ranks on realised tau, V (includes idiosyncratic noise, so no
X-measurable rule can reach it); feasible ranks on E[tau|X] E[V|X].  Welfare of
both is evaluated on realised tau, V.  Mirrors the seed sequence and split used
in simulation_v38.run_single_simulation, so numbers are comparable to Table 5.
Writes outputs/oracle_main.json.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from simulation_v38 import (
    N_PLAYERS, N_RUNS, SEED_BASE, ORACLE_MC, ORACLE_SEED_OFFSET,
)
from core.dgp import generate_main, conditional_means, FEATURE_NAMES
from core.io import save_json
from core.policy import (
    DEFAULT_BUDGET, compute_welfare, compute_risk_exposure,
    compute_mean_churn_moderation, target_topk, f_churn_vec,
)

NAMES = ('Oracle hindsight (g=1)', 'Oracle feasible (g=1)',
         'Oracle feasible (theta0)')


def run(n_runs=N_RUNS, n_players=N_PLAYERS):
    acc = {nm: {'welfare': [], 'risk': [], 'mod': [], 'ate': []} for nm in NAMES}
    for run_idx in range(n_runs):
        seed = SEED_BASE + run_idx * 7
        rng = np.random.RandomState(seed)
        data = generate_main(n_players, rng)

        _, test_idx = train_test_split(
            np.arange(len(data['treatment'])), test_size=0.30, random_state=seed)
        k = int(len(test_idx) * DEFAULT_BUDGET)

        true_tau = data['true_tau'][test_idx]
        true_ltv = data['true_ltv'][test_idx]
        true_churn = data['true_churn'][test_idx]
        y0, y1 = data['y0'][test_idx], data['y1'][test_idx]

        feats = {nm: data['features'][test_idx, i]
                 for i, nm in enumerate(FEATURE_NAMES)}
        cm = conditional_means('main', feats, n_mc=ORACLE_MC,
                               seed=seed + ORACLE_SEED_OFFSET)
        tv_bar = cm['tau_bar'] * cm['ltv_bar']
        scores = {
            NAMES[0]: true_tau * true_ltv,
            NAMES[1]: tv_bar,
            NAMES[2]: tv_bar * f_churn_vec(cm['churn_bar']),
        }
        for nm, sc in scores.items():
            targeted = target_topk(sc, k)
            a = acc[nm]
            a['welfare'].append(compute_welfare(targeted, true_tau, true_ltv))
            a['risk'].append(compute_risk_exposure(targeted, true_churn))
            a['mod'].append(compute_mean_churn_moderation(targeted, true_churn))
            a['ate'].append(float(np.mean(y1[targeted] - y0[targeted]) * 100))

    out = {'n_runs': n_runs, 'n_players': n_players, 'kappa': DEFAULT_BUDGET,
           'n_mc': ORACLE_MC, 'strategies': {}}
    print(f"tau*V oracles, kappa={DEFAULT_BUDGET}, N_RUNS={n_runs}")
    for nm, a in acc.items():
        w = np.array(a['welfare'])
        out['strategies'][nm] = {
            'welfare_mean': float(w.mean()), 'welfare_sd': float(w.std()),
            'ate_mean_pp': float(np.mean(a['ate'])), 'ate_sd_pp': float(np.std(a['ate'])),
            'risk_mean': float(np.mean(a['risk'])),
            'churn_mod_mean': float(np.mean(a['mod'])),
        }
        print(f"  {nm}: welfare {w.mean():.0f} +/- {w.std():.0f}, "
              f"ATE {np.mean(a['ate']):.1f} pp, risk {np.mean(a['risk']):.3f}")
    print("wrote", save_json(out, 'oracle_main.json'))
    return out


if __name__ == '__main__':
    run()
