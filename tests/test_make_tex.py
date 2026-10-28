"""make_tex: formatters, macro naming, generated-file contract, determinism, manuscript checks."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import make_tex as mt

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TEX = PAPER / "stent-2026-value-weighted-targeting.tex"
SMOKE = [ROOT / "outputs" / f"{b}_smoke.json" for b in mt.INPUT_BASENAMES]


_SMOKE_CACHE = {}


def _smoke_current():
    """True when every smoke JSON exists and has the schema this version of make_tex reads."""
    if not all(p.exists() for p in SMOKE):
        return False
    try:
        data = mt.load(ROOT / "outputs", True)
        mt.build(data)
    except (KeyError, ValueError):
        return False
    return True


def smoke_data():
    """Smoke JSONs, regenerated with run_all.py --smoke when missing or stale; fails (never
    skips) if they cannot be produced."""
    if "data" not in _SMOKE_CACHE:
        if not _smoke_current():
            proc = subprocess.run(
                [sys.executable, str(ROOT / "src" / "run_all.py"), "--smoke", "--jobs", "4"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
        assert _smoke_current(), "smoke outputs could not be regenerated"
        _SMOKE_CACHE["data"] = mt.load(ROOT / "outputs", True)
    return _SMOKE_CACHE["data"]


# ---- formatters


def test_thousands_separator():
    assert mt.fmt_welfare(1234567.4) == "1{,}234{,}567"
    assert mt.fmt_welfare(999.6) == "1{,}000"
    assert mt.fmt_welfare(-1234.2) == "-1{,}234"
    assert mt.fmt_welfare(12) == "12"
    assert mt.fmt_welfare(-0.2) == "0"


def test_precision_per_quantity():
    assert mt.fmt_harm(0.12345) == "0.123"
    assert mt.fmt_kappa(0.5) == "0.500"
    assert mt.fmt_d(1.4949) == "1.49"
    assert mt.fmt_pct(0.12345) == r"12.3\%"
    assert mt.fmt_pct(1.0291) == r"102.9\%"


def test_p_values():
    assert mt.fmt_p(0.0004) == "<0.001"
    assert mt.fmt_p(0.0009) == "<0.001"  # never printed as 0.001
    assert mt.fmt_p(0.00099999) == "<0.001"
    assert mt.fmt_p(0.0) == "<0.001"
    assert mt.fmt_p(0.001) == "0.0010"
    assert mt.fmt_p(0.0496) == "0.0496"  # 4 dp below 0.1, never rounds up to 0.050
    assert mt.fmt_p(0.0996) == "0.0996"
    assert mt.fmt_p(0.1) == "0.100"
    assert mt.fmt_p(0.5) == "0.500"
    assert mt.fmt_p(1.0) == "1.000"


def test_tex_makes_signs_safe():
    assert mt.tex("-1{,}234") == r"\ensuremath{-}1{,}234"
    assert mt.tex("<0.001") == r"\ensuremath{<}0.001"
    assert mt.tex("[-0.1, >0.500]") == r"[\ensuremath{-}0.1, \ensuremath{>}0.500]"
    assert mt.tex("T-learner") == "T-learner"


def test_signed_and_ci():
    assert mt.fmt_signed(5.2, mt.fmt_harm) == "+5.200"
    assert mt.fmt_signed(-5.2, mt.fmt_harm) == "-5.200"
    assert mt.fmt_ci(-1, 2000, mt.fmt_welfare) == "[-1, 2{,}000]"


def test_kappa_tags_spell_digits():
    assert mt.kappa_tag(0.1) == "Kten"
    assert mt.kappa_tag(0.3) == "Kthirty"
    assert mt.kappa_tag(0.5) == "Kfifty"
    assert mt.kappa_tag(0.25) == "Ktwentyfive"
    assert mt.kappa_tag(0.05) == "Kfive"
    assert mt.check_macro_name("numX" + mt.kappa_tag(0.075))  # fallback is letters only


def test_macro_name_validation():
    mt.check_macro_name("numMainWelfareCateOnlyKthirty")
    for bad in ("numMain2", "main", "num_x", "numlower", "numX1"):
        with pytest.raises(ValueError):
            mt.check_macro_name(bad)
    m = mt.Macros()
    m.add("numAbc", "1", "m", "s")
    with pytest.raises(ValueError):
        m.add("numAbc", "2", "m", "s")


def test_kappa_star_censoring():
    grid = [0.05, 0.5]
    base = {"point": None, "point_outcome": "above_grid", "ci95": [0.475, 0.5], "ci95_labels": [None, ">max"]}
    assert mt.kappa_star_text(base, grid, "point") == ">0.500"
    assert mt.kappa_star_text(base, grid, "lo") == "0.475"
    assert mt.kappa_star_text(base, grid, "hi") == ">0.500"
    below = dict(base, point_outcome="below_grid", ci95_labels=["<min", "<min"])
    assert mt.kappa_star_text(below, grid, "point") == "<0.050"
    assert mt.kappa_star_text(below, grid, "lo") == "<0.050"
    assert mt.kappa_star_text(dict(base, point=0.3), grid, "point") == "0.300"


# ---- generated files


def _data_rows(text):
    """Table-body lines that are data rows: no header, comments, rules, spacing or panel headings."""
    return [ln for ln in text.splitlines()[1:] if ln and not ln.startswith(("%",) + mt.STRUCTURAL_PREFIXES)]


def test_generated_files_contract():
    files = mt.build(smoke_data())
    assert set(files) == set(mt.generated_names())
    for name, text in files.items():
        assert text.startswith(mt.HEADER + "\n"), name
        assert "\u2014" not in text, name
    for name, text in files.items():
        if name.startswith("tab_"):
            assert "begin{table" not in text and "begin{tabular" not in text, name
            rows = [ln for ln in _data_rows(text)]
            assert rows and all(ln.endswith(r"\\") for ln in rows), name
            assert len({ln.count("&") for ln in rows}) == 1, f"{name}: ragged rows"
            assert not re.search(r"\bnan\b|None", text), name


def test_numbers_tex_documents_and_defines_every_macro():
    text = mt.build(smoke_data())["numbers.tex"]
    defined = re.findall(r"\\newcommand\{\\(num[A-Za-z]+)\}", text)
    documented = re.findall(r"^%   \\(num[A-Za-z]+): .+ \[.+\]$", text, flags=re.M)
    assert defined and defined == documented
    assert len(set(defined)) == len(defined)
    assert all(mt.MACRO_RE.match(n) for n in defined)


def _macros():
    return mt.build_macros(smoke_data()).items


def _num(text):
    """Numeric value of a formatted macro string, or None if censored/non-numeric."""
    t = text.replace("{,}", "").replace(r"\%", "").lstrip("+")
    try:
        return float(t)
    except ValueError:
        return None


def test_headline_macros_present():
    names = set(_macros())
    for need in (
        "numHarmExposureGapTauV",
        "numHarmExposureGapTauVCiLo",
        "numBudgetKappaStarPointTMisspec",
        "numMainEfficiencyLevelFullCalOracleFeasible",
        "numMainEfficiencyNormalizedFullCalOracleFeasibleCiHi",
        "numMainOracleHindsightWelfare",
        "numMainOracleFeasibleWelfare",
        "numLearnersDiffSTunedVsTTunedMisspecKthirty",
        "numLearnersShapleyCateSharePerRepTDefaultMisspec",
        "numLearnersShapleyCateShareRatioOfMeansTDefaultMisspec",
        "numRobustnessIdentityRateAipw",
        "numMainWelfareCateOnlyKthirty",
    ):
        assert need in names, need
    assert not any("Best" in n for n in names)  # no post-hoc argmax rule


def test_oracle_macros_are_labelled_feasible_or_hindsight():
    for n in _macros():
        assert not re.search(r"Oracle(?!Feasible|Hindsight)", n), n
        assert not re.search(r"(?<!Oracle)(Feasible|Hindsight)", n), n


def test_every_diff_macro_has_ci_and_p_siblings():
    names = set(_macros())
    diffs = [n for n in names if "Diff" in n and not re.search(r"(CiLo|CiHi|HolmP|P|D)$", n)]
    assert diffs
    for n in diffs:
        assert n + "CiLo" in names and n + "CiHi" in names, n
        assert n + "P" in names or n + "HolmP" in names, n


def test_every_point_estimate_lies_inside_its_ci():
    items = _macros()
    checked = 0
    for n, (value, _, _) in items.items():
        if n + "CiLo" not in items:
            continue
        pt, lo, hi = _num(value), _num(items[n + "CiLo"][0]), _num(items[n + "CiHi"][0])
        if None in (pt, lo, hi):
            continue
        assert lo <= pt <= hi, (n, lo, pt, hi)
        checked += 1
    assert checked > 40


def test_shapley_point_is_per_rep_mean_inside_its_ci():
    learners = smoke_data()["learners"]
    items = _macros()
    for combo, sp in learners["regret_split_kappa_0.30"].items():
        per = sp["shapley_tau_share_per_rep"]
        if per:
            assert per["ci95"][0] <= per["mean"] <= per["ci95"][1], combo
    n = "numLearnersShapleyCateSharePerRepTDefaultMisspec"
    per = learners["regret_split_kappa_0.30"]["T (default)|misspec"]["shapley_tau_share_per_rep"]
    assert items[n][0] == mt.fmt_pct(per["mean"])  # the point is the per-rep mean ...
    ratio = items["numLearnersShapleyCateShareRatioOfMeansTDefaultMisspec"]
    assert "ratio of mean" in ratio[1] and "no CI" in ratio[1]
    assert "numLearnersShapleyCateShareRatioOfMeansTDefaultMisspecCiLo" not in items  # ... the ratio has no CI


def test_efficiency_macros_come_from_stored_per_rep_ratios():
    data = smoke_data()
    e = data["main"]["efficiency"]["rules"]["Full Framework (Cal.)"]
    items = _macros()
    assert items["numMainEfficiencyLevelFullCalOracleFeasible"][0] == mt.fmt_pct(e["level_ratio"]["mean"])
    assert items["numMainEfficiencyNormalizedFullCalOracleFeasibleCiLo"][0] == mt.fmt_pct(e["normalized"]["ci95"][0])


def test_tab_main_cells_equal_formatted_json_values():
    data = smoke_data()
    main = data["main"]
    rows = [ln for ln in mt.build(data)["tab_main.tex"].splitlines()[1:] if ln != r"\midrule"]
    got = {r.split(" & ")[0]: r.rstrip(r" \\").split(" & ")[1:] for r in rows}

    def thousands(x):  # independent of make_tex's formatter
        return f"{round(x):,}".replace(",", "{,}")

    for name in mt.MAIN_ORDER + mt.ORACLES_ORDER:
        src = main["oracle_main"]["strategies"].get(name) or main["main_results"][name]
        w, sd, risk = src["welfare_mean"], src["welfare_sd"], src["risk_mean"]
        cells = got[mt.STRATEGY_LABEL[name]]
        assert cells == [thousands(w).replace("-", r"\ensuremath{-}"), thousands(sd), f"{risk:.3f}"], name


def test_main_results_are_stored_unrounded():
    w = smoke_data()["main"]["main_results"]["CATE Only"]["welfare_mean"]
    assert w != round(w)


def test_learner_lookups_check_the_baseline():
    rows = [
        {"combo": "S (tuned)|misspec", "baseline": "T (default)|misspec", "p_holm": 0.1},
        {"combo": "S (tuned)|misspec", "baseline": "T (tuned)|misspec", "p_holm": 0.2},
    ]
    assert mt.find_test(rows, "S (tuned)|misspec", "T (tuned)|misspec")["p_holm"] == 0.2
    assert mt.find_test(rows, "S (tuned)|misspec", "X (tuned)|misspec") is None
    with pytest.raises(ValueError):
        mt.find_test(rows + rows[:1], "S (tuned)|misspec", "T (default)|misspec")
    learners = smoke_data()["learners"]
    for rows in learners["tests_vs_T_default"].values():
        assert all(r["baseline"] == mt.default_baseline(r["combo"]) for r in rows)


def test_robustness_and_harm_table_content():
    data = smoke_data()
    files = mt.build(data)
    assert "pp" in files["tab_robustness.tex"].splitlines()[-1]  # obs_diagnostic unit
    assert "obs_diagnostic/None" not in files["numbers.tex"]
    assert "sigma_h by construction" in files["tab_harm.tex"]
    rows = _data_rows(files["tab_harm.tex"])
    assert len(rows) == 30 and rows[0].count("&") == 4  # 2 keys + 3 rules, panel A then panel B
    assert files["tab_harm.tex"].count("Panel ") == 2
    cal = files["tab_calibration.tex"]
    assert "degenerate" in cal and cal.count("--") >= 2


def test_make_tex_is_deterministic(tmp_path):
    data = smoke_data()
    assert mt.build(data) == mt.build(data)
    a, b = tmp_path / "a", tmp_path / "b"
    pa = mt.write(ROOT / "outputs", a, True)
    pb = mt.write(ROOT / "outputs", b, True)
    assert [p.name for p in pa] == [p.name for p in pb]
    for x, y in zip(pa, pb):
        assert x.read_bytes() == y.read_bytes()


def test_smoke_never_targets_tracked_dir():
    assert mt.SMOKE_GENERATED_DIR != mt.GENERATED_DIR
    assert mt.SMOKE_GENERATED_DIR.name == "generated_smoke"


# ---- manuscript checks (the LaTeX source is private: paper/ is absent in a fresh clone)

needs_paper = pytest.mark.skipif(
    not TEX.exists(), reason="paper/ is absent: the LaTeX source is private, so manuscript checks are skipped"
)


def _generated_numbers():
    p = PAPER / "generated" / "numbers.tex"
    assert p.exists(), "paper/generated/numbers.tex is missing: run python src/make_tex.py"
    return p.read_text()


@needs_paper
def test_all_macros_defined():
    defined = set(re.findall(r"\\newcommand\{\\(num[A-Za-z]+)\}", _generated_numbers()))
    assert defined, "numbers.tex defines no macros"
    used = set(re.findall(r"\\(num[A-Z][A-Za-z]*)", TEX.read_text()))
    assert used <= defined, f"undefined macros: {sorted(used - defined)}"


@needs_paper
def test_tracked_numbers_come_from_a_full_run():
    assert re.search(r"\\newcommand\{\\numMetaReps\}\{200\}", _generated_numbers())


@needs_paper
def test_no_hardcoded_table_numbers():
    text = TEX.read_text()
    bodies = [m.group(0) for m in re.finditer(r"\\begin\{(table\*?|tabular)\}.*?\\end\{\1\}", text, flags=re.S)]
    assert bodies and not [b for b in bodies if re.search(r"\d{1,2}\{,\}\d{3}", b)]


# ---- Task 10a: macro gaps G1-G17


def test_new_gap_macros_present():
    names = set(_macros())
    for need in (
        # G1, G2, G3
        "numHarmDiffHardCapMinusSoftGate",
        "numHarmDiffHardCapMinusSoftGateOracleFeasibleTauVHolmP",
        "numHarmDiffSoftGateMinusTauV",
        "numHarmDiffSoftGateMinusTauVOracleFeasibleTauVCiLo",
        "numHarmSafetyCostSoftGatePctCiHi",
        "numHarmSafetyCostHardCapOracleFeasibleTauVPct",
        "numHarmHardCapMinusSoftGateOracleFeasibleTauVPct",
        # G4, G5, G6
        "numHarmOvershootSoftGateSigmaHalfCiLo",
        "numHarmOvershootHardCapOracleFeasibleTauVSigmaOne",
        "numHarmViolationRateHardCapSigmaOne",
        "numHarmCorrHarmValueCMinusOne",
        "numHarmShareAboveSixTenthsCOne",
        "numHarmPopulationMeanCZero",
        # G7
        "numBudgetDiffCateLtvVsCateOnlyTMisspecKten",
        "numBudgetDiffCateLtvVsCateOnlyTMisspecKfiftyHolmP",
        "numBudgetDiffCateLtvVsCateOnlyOracleFeasibleOracleFeasibleKtenCiHi",
        # G8
        "numLearnersDiffEffectOnlyMinusTauVTTunedMisspecKtwenty",
        "numLearnersDiffEffectOnlyMinusTauVDRDefaultWellspecKfiftyHolmP",
        "numLearnersWelfareSTunedTauOnlyKtwenty",
        "numLearnersWelfareTDefaultMisspecKfifty",
        # G9-G11, G14, G16, G17
        "numRobustnessDiffSLearnerVsCateLtvMain",
        "numRobustnessDiffTunedMainHolmP",
        "numRobustnessDiffTunedAdversarialKten",
        "numRobustnessDiffLikeForLikeLoginSLearnerCiLo",
        "numRobustnessObsCateSharePerRepWellspecDefaultPlugin",
        "numBudgetBootstrapDraws",
        "numRobustnessDiffAipwCalibrationTunedP",
        # G12
        "numMainDefaultGPct",
        "numRobustnessLikeForLikeLoginCateOnlyPct",
        "numLearnersSTunedOverTTunedMisspecPctCiLo",
        "numRobustnessAdversarialKtenPct",
        "numRobustnessHeavyTailPct",
        "numRobustnessAipwCalibrationPctCiHi",
        # G15
        "numTuningGridSize",
        "numTuningPilotReps",
        "numTuningRLossT",
        "numTuningDepthS",
        "numTuningRLossSpreadLearners",
    ):
        assert need in names, need


def test_pct_macros_carry_a_ci_and_are_never_diff_named():
    items = _macros()
    pcts = [n for n in items if n.endswith("Pct")]
    assert len(pcts) >= 16
    for n in pcts:
        assert "Diff" not in n, n
        assert n + "CiLo" in items and n + "CiHi" in items, n
        if n.startswith(
            (
                "numHarm",
                "numMainDefaultG",
                "numLearnersSTuned",
                "numRobustnessLike",
                "numRobustnessAdv",
                "numRobustnessHeavy",
                "numRobustnessAipw",
            )
        ):
            assert "delta-method" in items[n][1], n
    # the stored ratio comes from the study's per-replication arrays, not from the macro layer
    rob = smoke_data()["robustness"]
    t = rob["heavy_tail"]["cells"]["heavy_tail"]["comparisons"]["CATE x LTV vs CATE Only"]
    assert items["numRobustnessHeavyTailPct"][0] == mt.fmt_signed(t["ratio_to_b"]["ratio"], mt.fmt_pct)
    assert items["numRobustnessHeavyTailPctCiLo"][0] == mt.fmt_signed(t["ratio_to_b"]["ci95"][0], mt.fmt_pct)
    ll = rob["like_for_like"]["cells"]["login_drop_main"]["comparisons"]["S-Learner: all features vs dropped"]
    assert items["numRobustnessLikeForLikeLoginSLearnerPct"][0] == mt.fmt_signed(ll["ratio_to_a"]["ratio"], mt.fmt_pct)


def test_harm_safety_cost_orientation_and_hard_cap_minus_soft_gate_sign():
    """cost = -(capped - blind)/blind: the CI is the stored ratio CI, flipped and sorted."""
    data = smoke_data()
    harm, items = data["harm_sweep"], _macros()
    t = mt.harm_test(harm, 1.0, 0.0, 0.3, "soft gate vs tauV", "estimated")
    r = t["ratio_to_blind"]
    assert items["numHarmSafetyCostSoftGatePct"][0] == mt.fmt_signed(-r["ratio"], mt.fmt_pct)
    assert items["numHarmSafetyCostSoftGatePctCiLo"][0] == mt.fmt_signed(-r["ci95"][1], mt.fmt_pct)
    assert "negative = the rule beats harm-blind" in items["numHarmSafetyCostSoftGatePct"][1]
    assert "negative = the rule beats harm-blind" in items["numHarmSafetyCostHardCapOracleFeasibleTauVPct"][1]
    # hard cap minus soft gate: positive exactly when the hard cap's welfare is higher
    for comps, stag, suffix in (("estimated", "", ""), ("oracle", "OracleFeasibleTauV", " (oracle tau,V)")):
        hard = mt.harm_cell(harm, 1.0, 0.0, 0.3, "hard cap" + suffix)["welfare"]["mean"]
        soft = mt.harm_cell(harm, 1.0, 0.0, 0.3, "soft gate" + suffix)["welfare"]["mean"]
        diff = items[f"numHarmDiffHardCapMinusSoftGate{stag}"][0]
        pct = items[f"numHarmHardCapMinusSoftGate{stag}Pct"][0]
        assert diff == mt.fmt_signed(hard - soft, mt.fmt_welfare)
        assert pct.startswith("+") == (hard > soft) and diff.startswith("+") == (hard > soft)
        assert "positive = the hard cap has higher welfare" in items[f"numHarmHardCapMinusSoftGate{stag}Pct"][1]
    assert not any("CapSaving" in n for n in items)


def test_g7_g10_g14_macros_equal_json_values():
    data = smoke_data()
    items = _macros()
    b2, rob = data["budget_2x2"], data["robustness"]
    # G7: T|misspec at 0.10 / 0.20 / 0.50 and oracle|oracle at 0.10
    for key, tag, kp, kt in (
        ("T|misspec", "TMisspec", 0.1, "Kten"),
        ("T|misspec", "TMisspec", 0.2, "Ktwenty"),
        ("T|misspec", "TMisspec", 0.5, "Kfifty"),
        ("oracle|oracle", "OracleFeasibleOracleFeasible", 0.1, "Kten"),
    ):
        e = mt.per_kappa_entry(b2, key, kp)
        n = f"numBudgetDiffCateLtvVsCateOnly{tag}{kt}"
        assert items[n][0] == mt.fmt_signed(e["diff"]["mean"], mt.fmt_welfare)
        assert items[n + "CiLo"][0] == mt.fmt_welfare(e["diff"]["ci95"][0])
        assert items[n + "HolmP"][0] == mt.fmt_p(e["p_holm"]) and items[n + "D"][0] == mt.fmt_d(e["cohens_d"])
    # G10: the tuned-T arm macros and the tuned table column
    files = mt.build(data)
    rows = _data_rows(files["tab_robustness.tex"])
    assert len(rows) == len(mt.ROBUSTNESS_ROWS)
    for i, (section, cell, label, comp_key) in enumerate(mt.ROBUSTNESS_ROWS):
        tc = mt.tuned_comparison(rob, section, cell, comp_key)
        if tc is None:
            assert rows[i].endswith("-- & -- \\\\"), rows[i]
            continue
        n = f"numRobustnessDiffTuned{mt.SCENARIO_TAG[(section, cell)]}"
        assert items[n][0] == mt.fmt_signed(tc["mean_diff"], mt.fmt_welfare)
        assert items[n + "HolmP"][0] == mt.fmt_p(tc["p_holm"])
        cell_text = mt.stacked(mt.fmt_signed(tc["mean_diff"], mt.fmt_welfare), mt.fmt_ci(*tc["ci_95"], mt.fmt_welfare))
        assert mt.tex(cell_text) in rows[i]
    # G14: CATE-share levels
    d = rob["obs_diagnostic"]["cells"]["wellspec"]["diagnostic"]["tuned"]["Oracle feasible"]
    assert items["numRobustnessObsCateSharePerRepWellspecTunedOracleFeasible"][0] == mt.fmt_pct(
        d["cate_share_per_rep"]["mean"] / 100.0
    )
    assert items["numRobustnessObsCateSharePerRepWellspecTunedOracleFeasibleCiLo"][0] == mt.fmt_pct(
        d["cate_share_per_rep"]["ci95"][0] / 100.0
    )
    assert items["numRobustnessObsCateShareRatioOfMeansWellspecTunedOracleFeasible"][0] == mt.fmt_pct(
        d["cate_share_ratio_of_means_pct"] / 100.0
    )
    assert "numRobustnessObsCateShareRatioOfMeansWellspecTunedOracleFeasibleCiLo" not in items
    d = rob["obs_diagnostic"]["cells"]["misspec"]["diagnostic"]["default"]["plug-in"]
    assert items["numRobustnessObsCateSharePerRepMisspecDefaultPlugin"][0] == mt.fmt_pct(
        d["cate_share_per_rep"]["mean"] / 100.0
    )
    assert not any(
        n.startswith("numRobustnessObsCateShare")
        and not n.startswith(("numRobustnessObsCateSharePerRep", "numRobustnessObsCateShareRatioOfMeans"))
        for n in items
    )


def test_tab_frontier_and_tuned_robustness_column():
    data = smoke_data()
    files = mt.build(data)
    rows = [r for r in files["tab_frontier.tex"].splitlines()[1:] if r != r"\midrule"]
    assert len(rows) == len(data["main"]["frontier"]) + 1 and rows[-1].startswith(r"Inverted $g$")
    assert all(r.count("&") == 2 for r in rows)
    rob = _data_rows(files["tab_robustness.tex"])
    assert all(r.count("&") == 6 for r in rob)  # label, diff/CI, d, Holm p, family, tuned diff/CI, tuned p
    assert rob[0].count("--") == 0 and "Main design" in rob[0]  # tuned arm exists for the main design
    assert rob[-1].count("--") >= 2  # obs_diagnostic has no tuned arm
    body = files["tab_robustness.tex"]
    assert body.count("Panel A") == 1 and body.count("Panel B") == 1
    assert body.index("Panel B") > body.index(rob[sum(c == mt.HEADLINE_KEY for *_, c in mt.ROBUSTNESS_ROWS) - 1])


def test_tuning_macros_match_frozen_file_and_learners_meta():
    data = smoke_data()
    tuned, learners, items = data["tuned_params"], data["learners"], _macros()
    for lr, res in tuned["learners"].items():
        assert res["params"] == learners["_meta"]["tuned_params"][lr]
        assert items[f"numTuningDepth{lr}"][0] == str(res["params"]["max_depth"])
    assert items["numTuningGridSize"][0] == str(len(tuned["_meta"]["grid"]))
    levels = [r["mean_loss"] for r in tuned["learners"].values()]
    assert items["numTuningRLossSpreadLearners"][0] == mt.fmt_fixed(max(levels) - min(levels), 5)


def test_holm_family_documented_for_new_tests():
    items = _macros()
    assert (
        "Holm family" in items["numHarmDiffHardCapMinusSoftGateHolmP"][1]
        or "Holm family" in items["numHarmDiffHardCapMinusSoftGate"][1]
    )
    assert "own effect-only family" in items["numLearnersDiffEffectOnlyMinusTauVTTunedMisspecKten"][1]


def test_decoupled_macros_table_and_siblings():
    data = smoke_data()
    harm, items = data["harm_sweep"], _macros()
    for base in (
        "numHarmDiffSoftGateMinusCateOnly",
        "numHarmDecoupledDiffSoftGateMinusCateOnly",
        "numHarmDecoupledDiffSoftGateMinusTauV",
        "numHarmDiffSoftGateMinusCateOnlyOracleFeasibleTauV",
        "numHarmDecoupledDiffSoftGateMinusCateOnlyOracleFeasibleTauV",
        "numHarmDecoupledDiffSoftGateMinusTauVOracleFeasibleTauV",
    ):
        for suffix in ("", "CiLo", "CiHi", "P", "HolmP", "D"):
            assert base + suffix in items, base + suffix
    for base in (
        "numHarmSoftGateMinusCateOnlyPct",
        "numHarmDecoupledSoftGateMinusCateOnlyPct",
        "numHarmDecoupledSafetyCostSoftGatePct",
        "numHarmSoftGateMinusCateOnlyOracleFeasibleTauVPct",
        "numHarmDecoupledSoftGateMinusCateOnlyOracleFeasibleTauVPct",
        "numHarmDecoupledSafetyCostSoftGateOracleFeasibleTauVPct",
    ):
        for suffix in ("", "CiLo", "CiHi"):
            assert base + suffix in items, base + suffix
    t = mt.harm_test(harm, 1.0, 0.0, 0.3, "soft gate vs CATE only", "estimated", "decoupled")
    assert items["numHarmDecoupledDiffSoftGateMinusCateOnly"][0] == mt.fmt_signed(t["mean_diff"], mt.fmt_welfare)
    assert t["family_size"] == 8
    for dtag, dgp in (("Coupled", "main"), ("Decoupled", "decoupled")):
        d = harm["_meta"]["coupling"][dgp]
        assert items[f"numHarmCoupling{dtag}PartialCorrTauHarmGivenVbar"][0] == mt.fmt_signed(
            d["partial_corr_tau_harm_given_Vbar"], lambda v: mt.fmt_fixed(v, 3)
        )
        assert f"numHarmCoupling{dtag}PartialCorrTauHarmGivenValue" in items
    rows = [r for r in mt.build(data)["tab_harm_decoupled.tex"].splitlines() if r and not r.startswith("%")]
    assert len(rows) == 2 * len(harm["_meta"]["sigma_h"])
    assert rows[0].startswith("coupled") and rows[1].startswith("decoupled")


def test_load_honours_outdir_for_non_full_tuned_params(tmp_path):
    import shutil

    smoke_data()  # make sure the smoke outputs exist
    n = mt.load(ROOT / "outputs", True)["learners"]["_meta"]["n_players"]
    assert n != 10000
    for p in SMOKE:
        shutil.copy(p, tmp_path / p.name)
    with pytest.raises(FileNotFoundError):  # tuned file is looked up in outdir, not outputs/
        mt.load(tmp_path, True)
    shutil.copy(ROOT / "outputs" / f"tuned_params_n{n}.json", tmp_path / f"tuned_params_n{n}.json")
    assert mt.load(tmp_path, True)["tuned_params"]["_meta"]["n_players"] == n


# ---- scope of the harm and value-weighting claims (kappa=0.10, CATE-only, decoupled oracle)


def test_harm_kten_and_budget_tables():
    data = smoke_data()
    harm, files = data["harm_sweep"], mt.build(data)
    rows = _data_rows(files["tab_harm_kten.tex"])
    assert len(rows) == 30 and rows[0].count("&") == 4
    em = mt.harm_cell(harm, 1.0, 0.0, 0.1, "tauV")["exposure_minus_pop"]
    assert mt.tex(mt.gap_text(em)) in files["tab_harm_kten.tex"]
    assert files["tab_harm_kten.tex"] != files["tab_harm.tex"]
    text = files["tab_harm_budget.tex"]
    dgps = harm["_meta"]["dgps"]
    data_rows = _data_rows(text)
    n_panel = len(mt.HARM_BUDGET_RULES) * len(dgps)
    assert len(data_rows) == 2 * n_panel  # panel A estimated, panel B feasible-oracle components
    assert text.count("Panel ") == 2
    assert all(r.count("&") == 3 for r in data_rows)
    # columns: kappa 0.10, 0.30; panel A estimated, panel B feasible-oracle
    for i, sfx in ((0, ""), (n_panel, mt.HARM_SUFFIX)):
        first = data_rows[i].rstrip(" \\").split(" & ")
        assert first[0] == mt.DGP_LABEL[dgps[0]] and "Harm-blind" in first[1]
        expect = [
            mt.tex(mt.gap_text(mt.harm_cell(harm, 1.0, 0.0, k, "tauV" + sfx, dgps[0])["exposure_minus_pop"]))
            for k in (0.1, 0.3)
        ]
        assert first[2:] == expect
    last = data_rows[-1].rstrip(" \\").split(" & ")
    assert last[0] == mt.DGP_LABEL[dgps[-1]] and last[1] == "CATE only"


def test_harm_scope_macros_equal_json_values():
    data = smoke_data()
    harm, items = data["harm_sweep"], _macros()
    for base, dgp, kappa, rule, field in mt.HARM_SCOPE_CELLS:
        cell = mt.harm_cell(harm, 1.0, 0.0, kappa, rule, dgp)
        if field == "gap":
            em = cell["exposure_minus_pop"]
            assert items[base][0] == mt.fmt_signed(em["mean"], mt.fmt_harm), base
            assert base + "CiLo" in items and base + "CiHi" in items, base
        elif field == "exposure":
            assert items[base][0] == mt.fmt_harm(cell["harm_exposure"]["mean"]), base
        else:
            assert items[base][0] == mt.fmt_welfare(cell["welfare"]["mean"]), base
    for c, kappa, tag in ((-1.0, 0.3, "CMinusOne"), (0.0, 0.3, "CZero"), (0.0, 0.1, "CZeroKten")):
        t = mt.harm_test(harm, c, 0.0, kappa, "soft gate vs tauV", "estimated")
        base = f"numHarmDiffSoftGateMinusTauV{tag}"
        assert items[base][0] == mt.fmt_signed(t["mean_diff"], mt.fmt_welfare)
        assert items[base + "HolmP"][0] == mt.fmt_p(t["p_holm"])
        assert "Holm family" in items[base][1]


def test_learner_count_and_effective_welfare_macros():
    data = smoke_data()
    learners, rob, items = data["learners"], data["robustness"], _macros()
    tests = learners["tests_effect_only_vs_tauV"][mt.key_str(0.5)]
    wins = sum(t["diff"] > 0 for t in tests)
    holm = sum(t["diff"] > 0 and t["p_holm"] < 0.05 for t in tests)
    assert items["numLearnersEffectOnlyCellsKfifty"][0] == str(len(tests))
    assert items["numLearnersEffectOnlyWinsKfifty"][0] == str(wins)
    assert items["numLearnersEffectOnlyWinsHolmKfifty"][0] == str(holm) and holm <= wins
    w = rob["effective_welfare"]["cells"]["main"]["welfare"]
    eff_only, eff_ltv = w["CATE Only [effective]"]["mean"], w["CATE x LTV [effective]"]["mean"]
    assert items["numRobustnessEffectiveWelfareCateOnly"][0] == mt.fmt_welfare(eff_only)
    assert items["numRobustnessEffectiveWelfareGapCateLtvMinusCateOnly"][0] == mt.fmt_signed(
        eff_ltv - eff_only, mt.fmt_welfare
    )
    assert "no CI" in items["numRobustnessEffectiveWelfareGapCateLtvMinusCateOnly"][1]
    base = "numRobustnessDiffOverstatementCateLtvMinusCateOnly"
    for suffix in ("", "CiLo", "CiHi", "HolmP"):
        assert base + suffix in items, base + suffix
