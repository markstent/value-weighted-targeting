"""Compute tau*V oracle welfare (g=1) for the main scenario at kappa=0.30.

Mirrors the seed sequence and split used in simulation_v38.run_single_simulation
for the main results, so the numbers are directly comparable to Table 5.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from simulation_v38 import (
    N_PLAYERS, N_RUNS, SEED_BASE,
)
from core.dgp import (
    generate_main,
)
from core.policy import (
    DEFAULT_BUDGET, compute_welfare, compute_risk_exposure, compute_mean_churn_moderation, target_topk,
)


def run():
    welfares = []
    risks = []
    mods = []
    ates = []
    for run_idx in range(N_RUNS):
        seed = SEED_BASE + run_idx * 7
        rng = np.random.RandomState(seed)
        data = generate_main(N_PLAYERS, rng)

        n = len(data['treatment'])
        indices = np.arange(n)
        _, test_idx = train_test_split(
            indices, test_size=0.30, random_state=seed)
        k = int(len(test_idx) * DEFAULT_BUDGET)

        true_tau = data['true_tau'][test_idx]
        true_ltv = data['true_ltv'][test_idx]
        true_churn = data['true_churn'][test_idx]
        y0 = data['y0'][test_idx]
        y1 = data['y1'][test_idx]

        scores = true_tau * true_ltv
        targeted = target_topk(scores, k)

        welfares.append(compute_welfare(targeted, true_tau, true_ltv))
        risks.append(compute_risk_exposure(targeted, true_churn))
        mods.append(compute_mean_churn_moderation(targeted, true_churn))
        ates.append(float(np.mean(y1[targeted] - y0[targeted]) * 100))

    welfares = np.array(welfares)
    ates = np.array(ates)
    print(f"tau*V Oracle (g=1), kappa=0.30, N_RUNS={N_RUNS}")
    print(f"  welfare mean = {welfares.mean():.0f}")
    print(f"  welfare sd   = {welfares.std():.0f}")
    print(f"  ATE mean (pp) = {ates.mean():.1f}")
    print(f"  ATE sd  (pp) = {ates.std():.1f}")
    print(f"  risk exposure mean = {np.mean(risks):.3f}")
    print(f"  churn mod mean     = {np.mean(mods):.3f}")


if __name__ == '__main__':
    run()
