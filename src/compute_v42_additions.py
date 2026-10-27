"""Revision-42 additions: the two substantive demonstrations from the second review.

  A. Component-wise update benefit. The decomposition's claimed advantage is that
     a single component can be improved in place. We demonstrate it: holding the
     T-Learner CATE estimate fixed, we swap the deliberately misspecified value
     model for a well-specified one and show welfare recovers. A joint model has
     no analogue (it must be retrained end-to-end). This puts a welfare number on
     the component-update property (auditability itself is a governance benefit,
     not a welfare quantity, and we do not claim otherwise).

  B. Hard harm constraint vs. soft harm weight. The soft multiplicative gate h can
     be overridden by a large tau*V. We instead solve the budget-and-harm-
     constrained problem
         max sum_{i in S} tau_i V_i   s.t. |S| = k and mean harm over S <= Hbar,
     whose KKT solution ranks by the penalised score tau*V - mu*harm with mu set
     so the harm constraint binds. At a matched harm-exposure level, the hard
     constraint achieves higher welfare than the soft gate, i.e. it sits on the
     welfare-harm efficient frontier the soft weight does not.

Usage:  python compute_v42_additions.py [N_RUNS]
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
    generate_main, generate_harm,
)
from core.estimators import (
    VALUE_STREAM_V42, seed_from,
    estimate_components,
)
from core.policy import (
    DEFAULT_BUDGET, target_topk, compute_welfare,
)
from compute_revision_additions import h_protection

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else N_RUNS_DEFAULT
SEEDS = [SEED_BASE + r * 7 for r in range(N_RUNS)]


def _wellspec_value_model(Xtr, Xte, seed):
    """A correctly-specified value model: trained on the true LTV functional form."""
    rng_v = np.random.RandomState(seed_from(seed, VALUE_STREAM_V42))
    # True DGP LTV form: 10*spend + 40*login + 5*age (cf. simulation_v38 eq for V).
    target = (10 * Xtr[:, 1] + 40 * Xtr[:, 2] + 5 * Xtr[:, 0]
              + rng_v.normal(0, 80, len(Xtr))).clip(50, 8000)
    m = GradientBoostingRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.1, random_state=0)
    m.fit(Xtr, target)
    return m.predict(Xte).clip(50, 8000)


def component_update():
    """Welfare of tauhat*Vhat with the misspecified vs a well-specified value model,
    holding tauhat fixed (the component-wise update the decomposition permits)."""
    mis, well = [], []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        data = generate_main(N_PLAYERS, rng)
        idx = np.arange(len(data['treatment']))
        tr, te = train_test_split(idx, test_size=0.30, random_state=seed)
        k = int(len(te) * DEFAULT_BUDGET)
        tau, V = data['true_tau'][te], data['true_ltv'][te]
        tauh, vh_mis, _ = estimate_components(data, tr, te, seed=seed)  # misspecified value
        vh_well = _wellspec_value_model(data['features'][tr], data['features'][te], seed)
        mis.append(compute_welfare(target_topk(tauh * vh_mis, k), tau, V))
        well.append(compute_welfare(target_topk(tauh * vh_well, k), tau, V))
    a, b = float(np.mean(mis)), float(np.mean(well))
    return {
        'welfare_misspecified_value': round(a, 0),
        'welfare_wellspecified_value': round(b, 0),
        'recovery_abs': round(b - a, 0),
        'recovery_pct': round(100 * (b - a) / a, 1),
    }


def _solve_mu(tauV, harm, k, target_harm):
    """Smallest mu>=0 such that mean harm of top-k by (tauV - mu*harm) <= target."""
    lo, hi = 0.0, float(tauV.max()) + 1.0
    # mean harm is (weakly) decreasing in mu; bisect.
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        sel = target_topk(tauV - mid * harm, k)
        if harm[sel].mean() > target_harm:
            lo = mid
        else:
            hi = mid
    return hi


def harm_constraint():
    """Compare soft-h gate vs hard harm constraint at matched harm exposure (oracle)."""
    soft_w, soft_h, hard_w, hard_h, ung_w, ung_h = [], [], [], [], [], []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        data = generate_harm(N_PLAYERS, rng)
        idx = np.arange(len(data['treatment']))
        tr, te = train_test_split(idx, test_size=0.30, random_state=seed)
        k = int(len(te) * DEFAULT_BUDGET)
        tau, V, harm = data['true_tau'][te], data['true_ltv'][te], data['true_harm'][te]
        tauV = tau * V

        # Unguarded
        s = target_topk(tauV, k)
        ung_w.append(compute_welfare(s, tau, V)); ung_h.append(float(harm[s].mean()))

        # Soft multiplicative gate h
        s = target_topk(tauV * h_protection(harm), k)
        sw = compute_welfare(s, tau, V); sh = float(harm[s].mean())
        soft_w.append(sw); soft_h.append(sh)

        # Hard constraint matched to the soft gate's harm exposure
        mu = _solve_mu(tauV, harm, k, sh)
        s = target_topk(tauV - mu * harm, k)
        hard_w.append(compute_welfare(s, tau, V)); hard_h.append(float(harm[s].mean()))

    sw, sh = float(np.mean(soft_w)), float(np.mean(soft_h))
    hw, hh = float(np.mean(hard_w)), float(np.mean(hard_h))
    uw, uh = float(np.mean(ung_w)), float(np.mean(ung_h))
    return {
        'unguarded': {'welfare': round(uw, 0), 'harm': round(uh, 3)},
        'soft_gate': {'welfare': round(sw, 0), 'harm': round(sh, 3)},
        'hard_constraint': {'welfare': round(hw, 0), 'harm': round(hh, 3)},
        'hard_minus_soft_welfare': round(hw - sw, 0),
        'hard_minus_soft_pct': round(100 * (hw - sw) / sw, 1),
    }


def main():
    print(f"Revision-42 additions | N_RUNS={N_RUNS}, kappa={DEFAULT_BUDGET}")
    out = {}

    print("A. Component-wise update benefit ...")
    out['component_update'] = component_update()
    c = out['component_update']
    print(f"   misspecified-V welfare {c['welfare_misspecified_value']:.0f} -> "
          f"well-specified-V welfare {c['welfare_wellspecified_value']:.0f} "
          f"(+{c['recovery_abs']:.0f}, +{c['recovery_pct']}%), tau-hat unchanged")

    print("B. Hard harm constraint vs soft gate (oracle, matched harm) ...")
    out['harm_constraint'] = harm_constraint()
    h = out['harm_constraint']
    print(f"   unguarded: W={h['unguarded']['welfare']:.0f} harm={h['unguarded']['harm']}")
    print(f"   soft gate: W={h['soft_gate']['welfare']:.0f} harm={h['soft_gate']['harm']}")
    print(f"   hard cons: W={h['hard_constraint']['welfare']:.0f} harm={h['hard_constraint']['harm']}")
    print(f"   hard - soft at matched harm: +{h['hard_minus_soft_welfare']:.0f} (+{h['hard_minus_soft_pct']}%)")

    out['_meta'] = {'n_runs': N_RUNS, 'kappa': DEFAULT_BUDGET, 'seeds_base': SEED_BASE}
    path = save_json(out, 'simulation_v42_additions.json')
    print(f"Saved {path}")


if __name__ == '__main__':
    main()
