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
    cells = [{"c_harm": 1.0, "sigma_h": s, "kappa": 0.30, "rule": r,
              "welfare": _ci(1000 + 10 * i), "harm_exposure": _ci(0.45 + 0.01 * i),
              "n_reps": 3}
             for s in SIGMA_H_GRID for i, r in enumerate(RULES)]
    cells.append({"c_harm": 0.0, "sigma_h": 0.0, "kappa": 0.10, "rule": "tauV",
                  "welfare": {"mean": 1.0, "ci95": None}, "harm_exposure": {"mean": 0.1, "ci95": None}})
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
             "S-Learner", "Oracle hindsight (theta0)", "Oracle feasible (g=1)", "Oracle hindsight (g=1)"]
    main = {"main_results": {k: {"welfare_mean": 1000.0 + i * 100, "welfare_sd": 50.0}
                             for i, k in enumerate(names)},
            "frontier": [{"config": "Calibrated", "risk_mean": 0.45, "welfare_mean": 1500.0},
                         {"config": "No reg (g=1)", "risk_mean": 0.47, "welfare_mean": 1600.0}]}
    (tmp_path / "main_smoke.json").write_text(json.dumps(main))
    (tmp_path / "harm_sweep_smoke.json").write_text(json.dumps(harm_json))
    paths = mf.build_all(tmp_path, tmp_path / "figs", smoke=True)
    assert sorted(p.name for p in paths) == sorted(mf.FIGURE_NAMES)
    assert all(p.exists() for p in paths)
