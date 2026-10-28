"""core.policy.misrank_terms: one exact telescoping order, shared by every study."""
import numpy as np

from core.policy import misrank_terms, target_topk


def test_telescoping_is_exact_in_one_order_with_no_dropped_cross_term():
    rng = np.random.RandomState(0)
    n, k = 400, 100
    tau_hat, tau_ref = rng.normal(0.05, 0.05, n), rng.normal(0.05, 0.05, n)
    v_hat, v_ref = rng.uniform(100, 900, n), rng.uniform(100, 900, n)
    cate, value, size = misrank_terms(tau_hat, v_hat, tau_ref, v_ref, k)
    dep = set(target_topk(tau_hat * v_hat, k))
    ref = set(target_topk(tau_ref * v_ref, k))
    m = np.array(sorted(dep ^ ref))
    assert size == len(m) > 0
    # the signed pieces sum exactly to the score gap, unit by unit
    signed = (tau_ref - tau_hat) * v_ref + tau_hat * (v_ref - v_hat)
    np.testing.assert_allclose(signed, tau_ref * v_ref - tau_hat * v_hat, atol=1e-9)
    # the reported terms are the means of the absolute pieces of exactly this split
    assert cate == np.mean(np.abs(tau_ref[m] - tau_hat[m]) * np.abs(v_ref[m]))
    assert value == np.mean(np.abs(tau_hat[m]) * np.abs(v_ref[m] - v_hat[m]))
    # triangle inequality: the terms bound the mean absolute score gap over M
    assert cate + value >= np.mean(np.abs(tau_ref[m] * v_ref[m] - tau_hat[m] * v_hat[m])) - 1e-9


def test_identical_scores_give_none():
    t, v = np.linspace(0, 1, 50), np.linspace(1, 2, 50)
    assert misrank_terms(t, v, t, v, 10) is None
