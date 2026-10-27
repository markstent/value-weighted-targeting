"""tau*V oracle welfare across the four budget levels in Table 8.

Same seed sequence and split as run_single_simulation for the main scenario,
so results are directly comparable to the published budget-sensitivity table.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from simulation_v38 import (
    N_PLAYERS, N_RUNS, BUDGET_FRACTIONS, SEED_BASE,
)
from core.dgp import (
    generate_main,
)
from core.policy import (
    compute_welfare,
)


def run():
    welfares = {kappa: [] for kappa in BUDGET_FRACTIONS}
    for run_idx in range(N_RUNS):
        seed = SEED_BASE + run_idx * 7
        rng = np.random.RandomState(seed)
        data = generate_main(N_PLAYERS, rng)

        n = len(data['treatment'])
        indices = np.arange(n)
        _, test_idx = train_test_split(
            indices, test_size=0.30, random_state=seed)

        true_tau = data['true_tau'][test_idx]
        true_ltv = data['true_ltv'][test_idx]
        scores = true_tau * true_ltv
        order = np.argsort(scores)

        for kappa in BUDGET_FRACTIONS:
            k = int(len(test_idx) * kappa)
            targeted = order[-k:]
            welfares[kappa].append(
                compute_welfare(targeted, true_tau, true_ltv))

    print(f"tau*V Oracle (g=1), N_RUNS={N_RUNS}")
    for kappa in BUDGET_FRACTIONS:
        arr = np.array(welfares[kappa])
        print(f"  kappa={kappa}: welfare = {arr.mean():.0f} +/- {arr.std():.0f}")


if __name__ == '__main__':
    run()
