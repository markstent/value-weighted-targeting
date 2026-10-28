"""make_figures: renders from the new JSON keys, distinct oracle colours, harm frontier panels."""
import json

import matplotlib
import pytest

import make_figures as mf
from study_harm import RULES, SIGMA_H_GRID


def _ci(m):
    return {"mean": m, "ci95": [m - 5, m + 5]}


@pytest.fixture
def harm_json():
    cells = [{"dgp": "main", "c_harm": 1.0, "sigma_h": s, "kappa": 0.30, "rule": r,
              "welfare": _ci(1000 + 10 * i), "harm_exposure": _ci(0.45 + 0.01 * i),
              "n_reps": 3}
             for s in SIGMA_H_GRID for i, r in enumerate(RULES)]
    cells.append({"dgp": "main", "c_harm": 0.0, "sigma_h": 0.0, "kappa": 0.10, "rule": "tauV",
                  "welfare": {"mean": 1.0, "ci95": None}, "harm_exposure": {"mean": 0.1, "ci95": None}})
    cells.append({"dgp": "decoupled", "c_harm": 1.0, "sigma_h": 0.0, "kappa": 0.30, "rule": "tauV",
                  "welfare": _ci(1.0), "harm_exposure": _ci(0.1), "n_reps": 3})   # must be ignored
    return {"cells": cells}


def test_harm_frontier_one_panel_per_sigma(tmp_path, harm_json, monkeypatch):
    seen = {}
    real = matplotlib.pyplot.close
    monkeypatch.setattr(matplotlib.pyplot, "close", lambda f: seen.setdefault("fig", f) and real(f))
    out = mf.fig_harm_frontier(harm_json, tmp_path)
    assert out.name == "fig_harm_frontier.pdf" and out.stat().st_size > 1000
    assert len(seen["fig"].axes) == len(SIGMA_H_GRID)


def test_harm_frontier_handles_missing_ci(tmp_path, harm_json):
    for c in harm_json["cells"]:
        c["welfare"]["ci95"] = None
    assert mf.fig_harm_frontier(harm_json, tmp_path).exists()


def test_oracle_bars_have_distinct_colours(tmp_path):
    import inspect
    src = inspect.getsource(mf.fig_welfare_bars)
    assert src.count("#1a5a1a") == 1
    assert '"Oracle feasible (g=1)":\n            colors.append' in src


def test_build_all_from_smoke_jsons(tmp_path, harm_json):
    names = ["Random", "CATE Only", "CATE x LTV", "Full Framework (Est.)", "Full Framework (Cal.)",
             "S-Learner", "Oracle feasible (g=1)", "Oracle hindsight (g=1)", "Oracle feasible (theta0)",
             "Oracle hindsight (theta0)"]
    main = {"main_results": {k: {"welfare_mean": 1000.0 + i * 100, "welfare_sd": 50.0}
                             for i, k in enumerate(names)},
            "frontier": [{"config": "Calibrated", "risk_mean": 0.45, "welfare_mean": 1500.0},
                         {"config": "No reg (g=1)", "risk_mean": 0.47, "welfare_mean": 1600.0}]}
    (tmp_path / "main_smoke.json").write_text(json.dumps(main))
    (tmp_path / "harm_sweep_smoke.json").write_text(json.dumps(harm_json))
    paths = mf.build_all(tmp_path, tmp_path / "figs", smoke=True)
    assert sorted(p.name for p in paths) == sorted(mf.FIGURE_NAMES)
    assert all(p.exists() for p in paths)


def test_harm_frontier_raises_on_missing_rule(tmp_path, harm_json):
    harm_json["cells"] = [c for c in harm_json["cells"] if c["rule"] != "hard cap (oracle tau,V)"]
    with pytest.raises(ValueError, match="hard cap \\(oracle tau,V\\)"):
        mf.fig_harm_frontier(harm_json, tmp_path)


def test_missing_sd_omits_error_bar(tmp_path):
    names = ["Random", "CATE Only", "CATE x LTV", "Full Framework (Est.)", "Full Framework (Cal.)",
             "S-Learner", "Oracle feasible (g=1)", "Oracle hindsight (g=1)", "Oracle feasible (theta0)",
             "Oracle hindsight (theta0)"]
    data = {"main_results": {k: {"welfare_mean": 1000.0 + 10 * i, "welfare_sd": None}
                             for i, k in enumerate(names)}}
    assert mf.fig_welfare_bars(data, tmp_path).exists()
    import inspect
    assert "or 0.0 for k in order" not in inspect.getsource(mf.fig_welfare_bars)


def test_welfare_bars_label_calibrated_and_pair_oracles():
    import inspect
    src = inspect.getsource(mf.fig_welfare_bars)
    assert r"Full (calibrated $g$, AIPW)" in src
    order = src.split("order = [")[1].split("]")[0]
    for pair in ("g=1", "theta0"):          # feasible immediately before hindsight, per pair
        assert order.index(f"Oracle feasible ({pair})") < order.index(f"Oracle hindsight ({pair})")
        assert order.index(f"Oracle hindsight ({pair})") - order.index(f"Oracle feasible ({pair})") < 40


def test_harm_frontier_welfare_axis_has_thousands_separators(tmp_path, harm_json, monkeypatch):
    seen = {}
    real = matplotlib.pyplot.close
    monkeypatch.setattr(matplotlib.pyplot, "close", lambda f: seen.setdefault("fig", f) and real(f))
    mf.fig_harm_frontier(harm_json, tmp_path)
    fmt = seen["fig"].axes[0].yaxis.get_major_formatter()
    assert fmt(12345.0, 0) == "12,345"


def test_frontier_title_refers_to_observation_1(tmp_path, monkeypatch):
    seen = {}
    real = matplotlib.pyplot.close
    monkeypatch.setattr(matplotlib.pyplot, "close", lambda f: seen.setdefault("fig", f) and real(f))
    mf.fig_frontier({"frontier": [{"config": "Calibrated", "risk_mean": 0.45, "welfare_mean": 1500.0}]}, tmp_path)
    assert "Observation 1" in seen["fig"].axes[0].get_title()
