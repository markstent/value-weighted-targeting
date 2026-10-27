"""Hindsight and feasible tau*V oracle welfare (g=1, theta0) across the four budgets.

Same seed sequence and split as run_single_simulation for the main scenario, so
results are comparable to the budget-sensitivity table.  Writes
outputs/oracle_budget.json.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from simulation_v38 import (
    N_PLAYERS, N_RUNS, BUDGET_FRACTIONS, SEED_BASE, ORACLE_MC, ORACLE_SEED_OFFSET,
)
from core.dgp import generate_main, conditional_means, FEATURE_NAMES
from core.io import save_json
from core.policy import compute_welfare, f_churn_vec

NAMES = ('Oracle hindsight (g=1)', 'Oracle feasible (g=1)',
         'Oracle feasible (theta0)')


def run(n_runs=N_RUNS, n_players=N_PLAYERS):
    welfares = {nm: {kappa: [] for kappa in BUDGET_FRACTIONS} for nm in NAMES}
    for run_idx in range(n_runs):
        seed = SEED_BASE + run_idx * 7
        rng = np.random.RandomState(seed)
        data = generate_main(n_players, rng)

        _, test_idx = train_test_split(
            np.arange(len(data['treatment'])), test_size=0.30, random_state=seed)

        true_tau = data['true_tau'][test_idx]
        true_ltv = data['true_ltv'][test_idx]
        feats = {nm: data['features'][test_idx, i]
                 for i, nm in enumerate(FEATURE_NAMES)}
        cm = conditional_means('main', feats, n_mc=ORACLE_MC,
                               seed=seed + ORACLE_SEED_OFFSET)
        tv_bar = cm['tau_bar'] * cm['ltv_bar']
        orders = {
            NAMES[0]: np.argsort(true_tau * true_ltv),
            NAMES[1]: np.argsort(tv_bar),
            NAMES[2]: np.argsort(tv_bar * f_churn_vec(cm['churn_bar'])),
        }
        for kappa in BUDGET_FRACTIONS:
            k = int(len(test_idx) * kappa)
            for nm, order in orders.items():
                welfares[nm][kappa].append(
                    compute_welfare(order[-k:], true_tau, true_ltv))

    out = {'n_runs': n_runs, 'n_players': n_players, 'n_mc': ORACLE_MC,
           'strategies': {}}
    print(f"tau*V oracles, N_RUNS={n_runs}")
    for nm in NAMES:
        out['strategies'][nm] = {}
        for kappa in BUDGET_FRACTIONS:
            arr = np.array(welfares[nm][kappa])
            out['strategies'][nm][str(kappa)] = {
                'welfare_mean': float(arr.mean()), 'welfare_sd': float(arr.std())}
            print(f"  {nm} kappa={kappa}: welfare = {arr.mean():.0f} +/- {arr.std():.0f}")
    print("wrote", save_json(out, 'oracle_budget.json'))
    return out


if __name__ == '__main__':
    run()
