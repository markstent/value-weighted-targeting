"""Depth-matched S-Learner / T-Learner ablation at kappa=0.30.

The published simulation runs the T-Learner at max_depth=3 and the S-Learner
at max_depth=4. This script runs the same 200 seeds with two depth-matched
variants so the welfare gap between joint and decomposed modelling can be
isolated from the tree-depth hyperparameter:

  - S-Learner at max_depth=3 (matches the T-Learner's depth)
  - T-Learner at max_depth=4 paired with the existing LTV model
    (matches the S-Learner's depth)
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from simulation_v38 import (
    N_PLAYERS, N_RUNS, SEED_BASE,
)
from core.dgp import (
    generate_main,
)
from core.estimators import (
    estimate_components,
)
from core.policy import (
    DEFAULT_BUDGET, compute_welfare, target_topk,
)


def tlearner_at_depth(X_train, X_test, W_train, Y_train, max_depth):
    treat_mask = W_train == 1
    ctrl_mask = W_train == 0
    model_t = GradientBoostingClassifier(
        n_estimators=100, max_depth=max_depth, learning_rate=0.1, random_state=0)
    model_c = GradientBoostingClassifier(
        n_estimators=100, max_depth=max_depth, learning_rate=0.1, random_state=0)
    model_t.fit(X_train[treat_mask], Y_train[treat_mask])
    model_c.fit(X_train[ctrl_mask], Y_train[ctrl_mask])
    p_t = model_t.predict_proba(X_test)[:, 1]
    p_c = model_c.predict_proba(X_test)[:, 1]
    return p_t - p_c


def slearner_at_depth(X_train, X_test, W_train, Y_train, ltv_hat, max_depth):
    X_aug = np.column_stack([X_train, W_train])
    slearner = GradientBoostingRegressor(
        n_estimators=100, max_depth=max_depth, learning_rate=0.1, random_state=0)
    slearner.fit(X_aug, Y_train)
    n_test = len(X_test)
    pred_t1 = slearner.predict(np.column_stack([X_test, np.ones(n_test)]))
    pred_t0 = slearner.predict(np.column_stack([X_test, np.zeros(n_test)]))
    tau_sl = pred_t1 - pred_t0
    return tau_sl * ltv_hat


def run():
    slearner3 = []
    tlearner4_ltv = []
    for run_idx in range(N_RUNS):
        seed = SEED_BASE + run_idx * 7
        rng = np.random.RandomState(seed)
        data = generate_main(N_PLAYERS, rng)

        n = len(data['treatment'])
        indices = np.arange(n)
        train_idx, test_idx = train_test_split(
            indices, test_size=0.30, random_state=seed)
        k = int(len(test_idx) * DEFAULT_BUDGET)

        true_tau = data['true_tau'][test_idx]
        true_ltv = data['true_ltv'][test_idx]

        X_train = data['features'][train_idx]
        X_test = data['features'][test_idx]
        W_train = data['treatment'][train_idx]
        Y_train = data['y_obs'][train_idx]

        # Shared value-model estimate (same misspecified LTV as the main run)
        _, ltv_hat, _ = estimate_components(data, train_idx, test_idx, seed=seed)

        # (a) S-Learner at depth 3
        sl3_scores = slearner_at_depth(
            X_train, X_test, W_train, Y_train, ltv_hat, max_depth=3)
        sl3_target = target_topk(sl3_scores, k)
        slearner3.append(compute_welfare(sl3_target, true_tau, true_ltv))

        # (b) T-Learner at depth 4 + LTV
        tau4 = tlearner_at_depth(
            X_train, X_test, W_train, Y_train, max_depth=4)
        tl4_target = target_topk(tau4 * ltv_hat, k)
        tlearner4_ltv.append(compute_welfare(tl4_target, true_tau, true_ltv))

    sl3 = np.array(slearner3)
    tl4 = np.array(tlearner4_ltv)
    print(f"N_RUNS = {N_RUNS}, kappa = {DEFAULT_BUDGET}")
    print(f"S-Learner (depth 3):            {sl3.mean():.0f} +/- {sl3.std():.0f}")
    print(f"T-Learner (depth 4) x LTV:      {tl4.mean():.0f} +/- {tl4.std():.0f}")
    print("  (Compare with published: S-Learner depth 4 = 24,310;")
    print("   CATE x LTV (T-Learner depth 3) = 16,427.)")


if __name__ == '__main__':
    run()
