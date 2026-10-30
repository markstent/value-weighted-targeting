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
    cells = [
        {
            "dgp": "main",
            "c_harm": 1.0,
            "sigma_h": s,
            "kappa": 0.30,
            "rule": r,
            "welfare": _ci(1000 + 10 * i),
            "harm_exposure": _ci(0.45 + 0.01 * i),
            "n_reps": 3,
        }
        for s in SIGMA_H_GRID
        for i, r in enumerate(RULES)
    ]
    cells.append(
        {
            "dgp": "main",
            "c_harm": 0.0,
            "sigma_h": 0.0,
            "kappa": 0.10,
            "rule": "tauV",
            "welfare": {"mean": 1.0, "ci95": None},
            "harm_exposure": {"mean": 0.1, "ci95": None},
        }
    )
    cells.append(
        {
            "dgp": "decoupled",
            "c_harm": 1.0,
            "sigma_h": 0.0,
            "kappa": 0.30,
            "rule": "tauV",
            "welfare": _ci(1.0),
            "harm_exposure": _ci(0.1),
            "n_reps": 3,
        }
    )  # must be ignored
    return {"cells": cells}


def test_harm_frontier_one_panel_per_sigma(tmp_path, harm_json, monkeypatch):
    seen = {}
    real = matplotlib.pyplot.close
    monkeypatch.setattr(matplotlib.pyplot, "close", lambda f: seen.setdefault("fig", f) and real(f))
    out = mf.fig_harm_frontier(harm_json, tmp_path)
    assert out.name == "fig_harm_frontier.pdf" and out.stat().st_size > 1000
    assert len(seen["fig"].axes) == len(SIGMA_H_GRID)
    # every panel shares one exposure range, so the panels line up
    assert len({ax.get_xlim() for ax in seen["fig"].axes}) == 1


def test_harm_frontier_handles_missing_ci(tmp_path, harm_json):
    for c in harm_json["cells"]:
        c["welfare"]["ci95"] = None
    assert mf.fig_harm_frontier(harm_json, tmp_path).exists()


def test_build_all_from_smoke_jsons(tmp_path, harm_json):
    main = {
        "frontier": [
            {"config": "Calibrated", "risk_mean": 0.45, "welfare_mean": 1500.0},
            {"config": "No reg (g=1)", "risk_mean": 0.47, "welfare_mean": 1600.0},
        ],
    }
    (tmp_path / "main_smoke.json").write_text(json.dumps(main))
    (tmp_path / "harm_sweep_smoke.json").write_text(json.dumps(harm_json))
    paths = mf.build_all(tmp_path, tmp_path / "figs", smoke=True)
    assert sorted(p.name for p in paths) == sorted(mf.FIGURE_NAMES)
    assert all(p.exists() for p in paths)


def test_harm_frontier_raises_on_missing_rule(tmp_path, harm_json):
    harm_json["cells"] = [c for c in harm_json["cells"] if c["rule"] != "hard cap (oracle tau,V)"]
    with pytest.raises(ValueError, match="hard cap \\(oracle tau,V\\)"):
        mf.fig_harm_frontier(harm_json, tmp_path)


def test_harm_frontier_welfare_axis_has_thousands_separators(tmp_path, harm_json, monkeypatch):
    seen = {}
    real = matplotlib.pyplot.close
    monkeypatch.setattr(matplotlib.pyplot, "close", lambda f: seen.setdefault("fig", f) and real(f))
    mf.fig_harm_frontier(harm_json, tmp_path)
    fmt = seen["fig"].axes[0].yaxis.get_major_formatter()
    assert fmt(12345.0, 0) == "12,345"


def test_frontier_has_no_title_and_no_undefined_notation(tmp_path, monkeypatch):
    seen = {}
    real = matplotlib.pyplot.close
    monkeypatch.setattr(matplotlib.pyplot, "close", lambda f: seen.setdefault("fig", f) and real(f))
    mf.fig_frontier({"frontier": [{"config": "Calibrated", "risk_mean": 0.45, "welfare_mean": 1500.0}]}, tmp_path)
    ax = seen["fig"].axes[0]
    assert ax.get_title() == ""  # the LaTeX caption carries the message
    assert "rho" not in ax.get_xlabel()
