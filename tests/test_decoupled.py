"""Decoupled harm DGP: harm depends on X only through V_bar, tau untouched; its feasible
oracle; coupling diagnostics; the soft-minus-CATE-only tests in every harm cell."""

import math

import numpy as np
import pytest

from core import dgp
from core.dgp import FEATURE_NAMES, conditional_means, draw_features
from core.stats import holm_bonferroni, paired_ratio_ci, paired_test_raw
from study_harm import coupling_diagnostics, simulate, summarise


def test_vbar_moments_match_a_large_draw():
    f = draw_features(400_000, np.random.RandomState(1))
    z = dgp.z_vbar(f)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1.0) < 0.01


def test_generate_harm_decoupled_changes_only_harm():
    main = dgp.generate_harm(500, np.random.RandomState(5), harm_rng=np.random.RandomState(1))
    dec = dgp.generate_harm(500, np.random.RandomState(5), harm_rng=np.random.RandomState(1), decoupled=True)
    np.testing.assert_array_equal(main["features"], dec["features"])
    for key in ("true_tau", "true_ltv", "true_churn", "treatment", "y_obs"):
        np.testing.assert_array_equal(main[key], dec[key])
    assert not np.allclose(main["true_harm"], dec["true_harm"])


def test_decoupled_harm_is_function_of_vbar_and_noise_only():
    f = draw_features(2000, np.random.RandomState(3))
    # same noise, two populations with the same V_bar but different spend/login/age
    f2 = dict(f)
    f2["weekly_spend"] = f["weekly_spend"] + 4.0
    f2["login_freq"] = f["login_freq"] - 1.0  # 10*4 - 40*1 = 0 change in V_bar
    np.testing.assert_allclose(dgp.z_vbar(f), dgp.z_vbar(f2), atol=1e-12)
    a = dgp.draw_harm_decoupled(f, np.random.RandomState(9))
    b = dgp.draw_harm_decoupled(f2, np.random.RandomState(9))
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert not np.allclose(dgp.draw_harm(f, np.random.RandomState(9)), dgp.draw_harm(f2, np.random.RandomState(9)))


def test_decoupled_marginal_logit_variance_matches_main():
    f = draw_features(200_000, np.random.RandomState(1))
    sds = []
    for draw in (dgp.draw_harm, dgp.draw_harm_decoupled):
        h = draw(f, np.random.RandomState(4), 1.0)
        sds.append(np.std(np.log(h / (1 - h))))
    assert abs(sds[0] - sds[1]) < 0.03


def test_decoupled_oracle_closed_form_noise_free():
    f = draw_features(800, np.random.RandomState(3))
    cm = conditional_means("harm_decoupled", f, n_mc=4, seed=1, noise_scale=0.0)
    expect = 1.0 / (1.0 + np.exp(-(-0.3 + dgp.HARM_SCALE * dgp.z_vbar(f))))
    np.testing.assert_allclose(cm["harm_bar"], expect, atol=1e-12)
    main = conditional_means("main", f, n_mc=4, seed=1, noise_scale=0.0)
    np.testing.assert_allclose(cm["tau_bar"], main["tau_bar"], atol=1e-12)  # tau unchanged


def test_decoupled_oracle_matches_brute_force_monte_carlo():
    f = draw_features(6, np.random.RandomState(4))
    cm = conditional_means("harm_decoupled", f, n_mc=2000, seed=7)
    n_rep = 4000
    rep = {k: np.repeat(v, n_rep) for k, v in f.items()}
    h = dgp.draw_harm_decoupled(rep, np.random.RandomState(99)).reshape(6, n_rep)
    tol = 5 * np.sqrt(cm["harm_se"] ** 2 + h.std(axis=1) ** 2 / n_rep)
    assert np.all(np.abs(cm["harm_bar"] - h.mean(axis=1)) < tol)


def test_partial_correlation_of_tau_and_harm_given_vbar():
    d = coupling_diagnostics()  # n = 200k, fixed seed
    assert abs(d["decoupled"]["partial_corr_tau_harm_given_Vbar"]) < 0.01
    assert d["main"]["partial_corr_tau_harm_given_Vbar"] > 0.15
    assert d["n"] == 200_000 and set(d) >= {"main", "decoupled"}
    for name in ("main", "decoupled"):
        assert 0.5 < d[name]["corr_harm_V"] < 1.0 and math.isfinite(d[name]["partial_corr_tau_harm_given_V"])


def test_simulate_decoupled_and_cate_only_tests_in_every_cell():
    kw = dict(reps=4, n_players=3000, c_harms=[1.0], sigma_hs=[0.0, 0.5], kappas=[0.30])
    for name in ("main", "decoupled"):
        raw = simulate(dgp=name, **kw)
        s = summarise(raw, n_boot=50, dgp=name)
        assert {c["dgp"] for c in s["cells"]} == {name}
        tests = s["paired_tests"]
        assert {t["dgp"] for t in tests} == {name}
        assert len(tests) == 2 * 8  # 2 sigma cells x (6 + 2)
        for sigma in (0.0, 0.5):
            fam = [t for t in tests if t["sigma_h"] == sigma]
            assert len(fam) == 8 and all(t["family_size"] == 8 for t in fam)
            assert [t["p_holm"] for t in fam] == list(holm_bonferroni([t["p_raw"] for t in fam]))
            new = {t["components"]: t for t in fam if t["comparison"] == "soft gate vs CATE only"}
            assert set(new) == {"estimated", "oracle"}
            for comp, suffix in (("estimated", ""), ("oracle", " (oracle tau,V)")):
                t = new[comp]
                wa = np.array(raw[(1.0, sigma, 0.30, "soft gate" + suffix)]["welfare"])
                wb = np.array(raw[(1.0, sigma, 0.30, "CATE only" + suffix)]["welfare"])
                ref = paired_test_raw(wa, wb)
                assert t["mean_diff"] == ref["mean_diff"] and t["ci95"] == ref["ci_95"]
                r, lo, hi = paired_ratio_ci(wa - wb, wb)
                assert t["ratio_to_cate_only"] == {"ratio": r, "ci95": [lo, hi]}


def test_simulate_rejects_unknown_dgp():
    with pytest.raises(ValueError):
        simulate(reps=1, n_players=500, dgp="nope")
    assert FEATURE_NAMES
