"""Order-free (Shapley) regret split between CATE error and value-model error."""

import numpy as np

from core.policy import regret_split


def _world(seed=0, n=2000):
    rng = np.random.RandomState(seed)
    tau_true = rng.uniform(0, 0.3, n)
    ltv_true = rng.uniform(100, 1000, n)
    tau_ref = tau_true + rng.normal(0, 0.01, n)
    ltv_ref = ltv_true + rng.normal(0, 30, n)
    tau_hat = tau_ref + rng.normal(0, 0.08, n)
    ltv_hat = ltv_ref + rng.normal(0, 250, n)
    return tau_ref, ltv_ref, tau_hat, ltv_hat, tau_true, ltv_true


def test_regret_shapley_sums():
    tr, lr, th, lh, tt, lt = _world()
    r = regret_split(tr, lr, th, lh, tt, lt, k=600)
    for key in ("order_a", "order_b", "shapley"):
        assert abs(r[key]["tau_share"] + r[key]["v_share"] - 1) < 1e-12
    assert abs(r["phi_tau"] + r["phi_v"] - r["total"]) < 1e-9
    assert abs(r["shapley"]["tau_share"] - 0.5 * (r["order_a"]["tau_share"] + r["order_b"]["tau_share"])) < 1e-12
    assert not r["zero_denominator"]


def test_value_model_equals_reference_gives_all_cate():
    tr, lr, th, _, tt, lt = _world(1)
    r = regret_split(tr, lr, th, lr, tt, lt, k=600)
    assert r["phi_v"] == 0
    assert r["shapley"]["tau_share"] == 1
    assert r["order_a"]["tau_share"] == 1 and r["order_b"]["tau_share"] == 1


def test_order_matters_for_single_paths_but_shapley_is_symmetric():
    tr, lr, th, lh, tt, lt = _world(2)
    r = regret_split(tr, lr, th, lh, tt, lt, k=600)
    # Swapping which component is called "tau" must swap the Shapley shares.
    r_swapped = regret_split(lr, tr, lh, th, lt, tt, k=600)
    assert abs(r["shapley"]["tau_share"] - r_swapped["shapley"]["v_share"]) < 1e-12


def test_zero_denominator_flagged_nan():
    tr, lr, _, _, tt, lt = _world(3)
    r = regret_split(tr, lr, tr, lr, tt, lt, k=600)
    assert r["zero_denominator"]
    assert np.isnan(r["shapley"]["tau_share"]) and np.isnan(r["shapley"]["v_share"])
