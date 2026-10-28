"""core.policy.misrank_terms: the terms bound the mean absolute score gap over the misranked set
in one fixed telescoping order (not a regret decomposition)."""

import numpy as np
import pytest

from core.policy import misrank_terms, target_topk


def _setup(seed, n=400):
    rng = np.random.RandomState(seed)
    return (rng.normal(0.05, 0.05, n), rng.uniform(100, 900, n), rng.normal(0.05, 0.05, n), rng.uniform(100, 900, n))


def _misranked(tau_hat, v_hat, tau_ref, v_ref, k):
    return np.array(sorted(set(target_topk(tau_hat * v_hat, k)) ^ set(target_topk(tau_ref * v_ref, k))))


@pytest.mark.parametrize("seed", range(5))
def test_terms_bound_the_mean_absolute_score_gap_and_match_hand_formula(seed):
    tau_hat, v_hat, tau_ref, v_ref = _setup(seed)
    cate, value, size = misrank_terms(tau_hat, v_hat, tau_ref, v_ref, 100)
    m = _misranked(tau_hat, v_hat, tau_ref, v_ref, 100)
    assert size == len(m) > 0
    gap = np.mean(np.abs(tau_ref[m] * v_ref[m] - tau_hat[m] * v_hat[m]))
    assert cate + value >= gap - 1e-9  # the bound property
    assert cate == pytest.approx(np.mean(np.abs(tau_ref[m] - tau_hat[m]) * np.abs(v_ref[m])))
    assert value == pytest.approx(np.mean(np.abs(tau_hat[m]) * np.abs(v_ref[m] - v_hat[m])))


def test_bound_is_tight_when_only_one_component_is_wrong():
    tau_hat, v_hat, tau_ref, v_ref = _setup(7)
    m = _misranked(tau_hat, v_ref, tau_ref, v_ref, 100)  # same V: only CATE wrong
    cate, value, _ = misrank_terms(tau_hat, v_ref, tau_ref, v_ref, 100)
    assert value == 0.0
    assert cate == pytest.approx(np.mean(np.abs(tau_ref[m] * v_ref[m] - tau_hat[m] * v_ref[m])))
    m = _misranked(tau_ref, v_hat, tau_ref, v_ref, 100)  # same tau: only V wrong
    cate, value, _ = misrank_terms(tau_ref, v_hat, tau_ref, v_ref, 100)
    assert cate == 0.0
    assert value == pytest.approx(np.mean(np.abs(tau_ref[m] * v_ref[m] - tau_ref[m] * v_hat[m])))


def test_order_is_cate_at_reference_v_then_value_at_deployed_tau():
    tau_hat, v_hat = np.array([1.0, 0.0]), np.array([1.0, 1.0])
    tau_ref, v_ref = np.array([0.0, 1.0]), np.array([2.0, 3.0])
    cate, value, size = misrank_terms(tau_hat, v_hat, tau_ref, v_ref, 1)
    assert size == 2
    assert cate == pytest.approx((abs(0 - 1) * 2 + abs(1 - 0) * 3) / 2)  # |dtau| * |v_ref|
    assert value == pytest.approx((abs(1) * abs(2 - 1) + 0.0) / 2)  # |tau_hat| * |dv|


def test_identical_scores_give_none():
    t, v = np.linspace(0, 1, 50), np.linspace(1, 2, 50)
    assert misrank_terms(t, v, t, v, 10) is None
