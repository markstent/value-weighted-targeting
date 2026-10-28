"""study_main: oracle tables folded in, ddof=1 descriptive SDs, --smoke wiring, no legacy names."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import study_main
from core.io import load_json

ROOT = Path(__file__).resolve().parents[1]


def test_sd_is_sample_sd():
    x = [1.0, 2.0, 4.0, 8.0]
    assert study_main._sd(x) == pytest.approx(np.std(x, ddof=1))
    assert study_main._sd(x, 1) == round(np.std(x, ddof=1), 1)
    assert study_main._sd([3.0]) is None


def test_aggregate_uses_ddof1():
    runs = [
        {"ate": a, "welfare": w, "risk_exposure": 0.5, "mean_churn_mod": 0.1, "mean_ltv_cr": 1.0}
        for a, w in [(1, 1000), (2, 1500), (4, 4000)]
    ]
    agg = study_main._aggregate_results({"S": runs}, ["S"])["S"]
    assert agg["welfare_sd"] == pytest.approx(np.std([1000, 1500, 4000], ddof=1))
    assert agg["welfare_mean"] == pytest.approx(2166.6666666666665)  # stored unrounded


def test_efficiency_is_mean_of_per_rep_ratios():
    def runs(ws):
        return [{"welfare": float(w)} for w in ws]

    res = {
        "Random": runs([0, 100]),
        "Oracle feasible (g=1)": runs([1000, 400]),
        "Oracle hindsight (g=1)": runs([2000, 800]),
    }
    res.update({r: runs([500, 300]) for r in study_main.EFFICIENCY_RULES})
    eff = study_main._efficiency_table(res, 2, 100)
    rule = eff["rules"]["Full Framework (Cal.)"]
    assert rule["level_ratio"]["mean"] == pytest.approx((0.5 + 0.75) / 2)  # not 800/1400
    assert rule["normalized"]["mean"] == pytest.approx((0.5 + 200 / 300) / 2)
    lo, hi = rule["normalized"]["ci95"]
    assert lo < rule["normalized"]["mean"] < hi
    assert eff["feasible_over_hindsight"]["mean"] == pytest.approx(0.5)
    assert eff["oracle"] == "Oracle feasible (g=1)"


def _runs(ws):
    return [{"welfare": float(w)} for w in ws]


def test_budget_p_values_are_unrounded_and_reach_make_tex():
    import make_tex

    rng = np.random.RandomState(1)
    base = rng.normal(1000, 50, 12)
    noise = rng.normal(0, 1, 12)
    strat = {
        "Full Framework (Cal.)": _runs(base + 30 + noise),
        "CATE x LTV": _runs(base),
        "CATE Only": _runs(base + 29 + rng.normal(0, 1, 12)),
    }
    bp = study_main._budget_paired(strat, "0.3")
    t = bp["cal_vs_ltv"]
    assert 0 < t["p_value"] < 0.0005 and t["p_holm"] >= t["p_value"]
    assert t["p_value"] != round(t["p_value"], 4)  # not rounded to 4 dp
    # a Holm p in [0.0005, 0.001) would have printed 0.0010 after rounding to 4 dp
    low = dict(t, p_holm=0.00096)
    assert make_tex.fmt_p(low["p_holm"]) == "<0.001"
    m = make_tex.Macros()
    make_tex.add_test(m, "numTestDiff", low, make_tex.fmt_welfare, "x", "s")
    assert m.items["numTestDiffHolmP"][0] == "<0.001"


def test_efficiency_exclusion_is_a_hard_error():
    res = {
        "Random": _runs([0, 100]),
        "Oracle feasible (g=1)": _runs([1000, 50]),
        "Oracle hindsight (g=1)": _runs([2000, 800]),
    }
    res.update({r: _runs([500, 300]) for r in study_main.EFFICIENCY_RULES})
    with pytest.raises(ValueError, match="no replication may be dropped"):
        study_main._efficiency_table(res, 2, 100)


def test_oracle_tables_match_budget_runs():
    kappa = "0.3"
    res = {
        nm: [{"welfare": float(w), "ate": 1.0, "risk_exposure": 0.4, "mean_churn_mod": 0.2} for w in (10, 20, 40)]
        for nm in study_main.ORACLE_STRATEGIES
    }
    main_t = study_main._oracle_table(res, 3, 100)
    bud_t = study_main._oracle_budget_table({kappa: res}, 3, 100)
    for nm in study_main.ORACLE_STRATEGIES:
        assert main_t["strategies"][nm]["welfare_sd"] == pytest.approx(np.std([10, 20, 40], ddof=1))
        assert bud_t["strategies"][nm][kappa]["welfare_mean"] == pytest.approx(70 / 3)


def test_smoke_cli_writes_main_json_with_oracles():
    subprocess.run([sys.executable, str(ROOT / "src" / "study_main.py"), "--smoke"], check=True, capture_output=True)
    out = load_json("main_smoke.json")
    assert out["design"]["n_runs"] == 3 and out["design"]["n_players"] == 2000
    assert set(out["oracle_main"]["strategies"]) == set(study_main.ORACLE_STRATEGIES)
    for kappa in study_main.BUDGET_FRACTIONS:
        for nm in study_main.ORACLE_STRATEGIES:
            assert str(kappa) in out["oracle_budget"]["strategies"][nm]
    # oracle_main at kappa=0.30 equals the budget table entry (same runs)
    for nm in study_main.ORACLE_STRATEGIES:
        assert out["oracle_main"]["strategies"][nm]["welfare_mean"] == pytest.approx(
            out["oracle_budget"]["strategies"][nm]["0.3"]["welfare_mean"]
        )
    assert "tost_cal_vs_cate" in out["cate_misspec_paired"]
    w = out["main_results"]["CATE Only"]["welfare_mean"]
    assert w != round(w)  # stored unrounded
    for rule in study_main.EFFICIENCY_RULES:
        e = out["efficiency"]["rules"][rule]
        for k in ("level_ratio", "normalized"):
            assert e[k]["ci95"][0] <= e[k]["mean"] <= e[k]["ci95"][1]


def test_no_legacy_version_names():
    pat = re.compile(r"v29|v38|v39|v4\d")
    bad = []
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").glob("*.py")) + [ROOT / "README.md"]:
        if path == Path(__file__):
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if pat.search(line):
                bad.append(f"{path.name}:{i}")
    assert not bad, bad
    assert not list((ROOT / "src").glob("*v[0-9]*")) and not (ROOT / "src" / "compute_tauv_oracle.py").exists()
