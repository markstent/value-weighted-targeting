"""Revision-45 addition: like-for-like joint-model comparison (review, Major #4).

The published misspecification result drops a feature from the decomposed
T-Learner's CATE model but leaves the S-Learner with all four features, then
concludes the joint model is "more robust". That comparison is not like-for-like:
the S-Learner is simply better-informed. This script re-runs the S-Learner with
the SAME feature dropped from its outcome model, in both the published (login-drop)
and adversarial (bonus_history-drop) misspecifications, to test whether the joint
model degrades too. If it does, the correct framing is "architectural advantage,
better-informed here" rather than "more robust to component misspecification".

Same 200 seeds (SEED_BASE + 7r) as the published study; kappa = 0.30.

Usage:  python compute_v45_additions.py [N_RUNS]
"""

import sys
import numpy as np
from sklearn.model_selection import train_test_split

from simulation_v38 import (
    N_PLAYERS, N_RUNS as N_RUNS_DEFAULT, SEED_BASE,
)
from core.io import save_json
from core.dgp import (
    generate_main, generate_adv,
)
from core.estimators import (
    estimate_components, _estimate_slearner,
)
from core.policy import (
    target_topk, compute_welfare,
)
from core.stats import (
    paired_test,
)

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else N_RUNS_DEFAULT
SEEDS = [SEED_BASE + r * 7 for r in range(N_RUNS)]
KAPPA = 0.30


def _welfare(scores, k, true_tau, true_ltv):
    return compute_welfare(target_topk(scores, k), true_tau, true_ltv)


def run_like_for_like(dgp_func, drop_idx, label):
    """S-Learner with all features vs S-Learner missing feature `drop_idx`.

    ltv_hat is the full-feature value estimate in both cases (matching the
    decomposed misspecification, which drops the feature only from the CATE
    model and keeps the value model intact).
    """
    keep_cols = [c for c in range(4) if c != drop_idx]
    sl_full, sl_drop, cate_full = [], [], []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        data = dgp_func(N_PLAYERS, rng)
        n = len(data['treatment'])
        train_idx, test_idx = train_test_split(np.arange(n), test_size=0.30,
                                               random_state=seed)
        tau_hat, ltv_hat, _ = estimate_components(data, train_idx, test_idx, seed=seed)
        true_tau = data['true_tau'][test_idx]
        true_ltv = data['true_ltv'][test_idx]
        k = int(len(test_idx) * KAPPA)

        Xtr, Xte = data['features'][train_idx], data['features'][test_idx]
        Wtr, Ytr = data['treatment'][train_idx], data['y_obs'][train_idx]

        # full-feature S-Learner (the published comparator)
        s_full = _estimate_slearner(Xtr, Xte, Wtr, Ytr, ltv_hat)
        # like-for-like S-Learner: same feature dropped from its outcome model
        s_drop = _estimate_slearner(Xtr[:, keep_cols], Xte[:, keep_cols],
                                    Wtr, Ytr, ltv_hat)

        sl_full.append(_welfare(s_full, k, true_tau, true_ltv))
        sl_drop.append(_welfare(s_drop, k, true_tau, true_ltv))
        cate_full.append(_welfare(tau_hat, k, true_tau, true_ltv))

    sl_full = np.array(sl_full)
    sl_drop = np.array(sl_drop)
    pt = paired_test(list(sl_drop), list(sl_full), f'{label} sl_drop vs full')
    return {
        'label': label,
        'dropped_feature_index': drop_idx,
        's_learner_full': round(float(sl_full.mean()), 0),
        's_learner_full_sd': round(float(sl_full.std()), 0),
        's_learner_dropped': round(float(sl_drop.mean()), 0),
        's_learner_dropped_sd': round(float(sl_drop.std()), 0),
        'degradation_abs': round(float(sl_full.mean() - sl_drop.mean()), 0),
        'degradation_pct': round(100 * (sl_full.mean() - sl_drop.mean()) / sl_full.mean(), 1),
        'paired_p': pt['p_value'],
        'cohens_d': pt['cohens_d'],
    }


def main():
    print("=" * 70)
    print(f"v45 like-for-like joint-model comparison (N_RUNS={N_RUNS}, kappa={KAPPA})")
    print("=" * 70)
    out = {}
    # published misspecification: drop login (index 2)
    out['login_drop_main'] = run_like_for_like(generate_main, 2, 'login-drop (main DGP)')
    # adversarial misspecification: drop bonus_history (index 3)
    out['bonus_drop_adversarial'] = run_like_for_like(generate_adv, 3, 'bonus-drop (adversarial DGP)')
    out['meta'] = {'n_runs': N_RUNS, 'kappa': KAPPA, 'seeds': 'SEED_BASE + 7r'}

    for key in ['login_drop_main', 'bonus_drop_adversarial']:
        r = out[key]
        print(f"\n{r['label']}:")
        print(f"  S-Learner full features : {r['s_learner_full']:,.0f}")
        print(f"  S-Learner dropped feat  : {r['s_learner_dropped']:,.0f}")
        print(f"  degradation             : {r['degradation_abs']:,.0f} "
              f"({r['degradation_pct']}%), p={r['paired_p']}, d={r['cohens_d']}")

    path = save_json(out, 'simulation_v45_additions.json')
    print(f"\nSaved to {path}")


if __name__ == '__main__':
    main()
