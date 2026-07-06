"""Generate the three figures referenced in manuscript_v39.tex.

Reads outputs/simulation_v29.json for empirical numbers. Produces vector PDFs
in docs/figures/. Re-run with:

    python src/figures_v39.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "outputs" / "simulation_v29.json"
FIGDIR = REPO / "docs" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )


def g_piecewise(p: np.ndarray, theta: tuple[float, float, float, float, float]) -> np.ndarray:
    p_L, p_ML, p_MH, p_H, phi = theta
    out = np.zeros_like(p, dtype=float)
    mask_ramp_up = (p >= p_L) & (p < p_ML)
    mask_plateau = (p >= p_ML) & (p <= p_MH)
    mask_ramp_down = (p > p_MH) & (p < p_H)
    mask_floor = p >= p_H
    out[mask_ramp_up] = (p[mask_ramp_up] - p_L) / (p_ML - p_L)
    out[mask_plateau] = 1.0
    out[mask_ramp_down] = 1.0 - (1.0 - phi) * (p[mask_ramp_down] - p_MH) / (p_H - p_MH)
    out[mask_floor] = phi
    return out


def fig_g_function() -> Path:
    theta = (0.05, 0.40, 0.60, 0.90, 0.20)
    p_L, p_ML, p_MH, p_H, phi = theta
    p = np.linspace(0, 1, 1000)
    g = g_piecewise(p, theta)

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(p, g, color="#1f3b5c", linewidth=1.8)
    ax.fill_between(p, 0, g, color="#1f3b5c", alpha=0.08)

    breakpoints = [(p_L, "$p_L$"), (p_ML, "$p_{ML}$"), (p_MH, "$p_{MH}$"), (p_H, "$p_H$")]
    for x, label in breakpoints:
        ax.axvline(x, color="#888", linestyle=":", linewidth=0.7)
        ax.annotate(
            label,
            xy=(x, 0),
            xytext=(x, -0.09),
            ha="center",
            va="top",
            fontsize=9,
        )

    ax.axhline(phi, color="#b34", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.annotate(r"$\phi$", xy=(1.01, phi), ha="left", va="center", fontsize=9, color="#b34")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.1)
    # Drop the numeric ticks at 0.4 and 0.6: they overprint the p_ML / p_MH
    # breakpoint labels below the axis. The breakpoint symbols mark those
    # positions, and the caption gives the theta values.
    ax.set_xticks([0.0, 0.2, 0.8, 1.0])
    ax.set_xlabel("churn probability $p$")
    ax.set_ylabel(r"risk-modulation weight $g(p;\theta)$")
    ax.set_title(r"Piecewise linear $g$ with default $\theta=(0.05,0.40,0.60,0.90,0.20)$", pad=10)

    regimes = [
        (0.0, p_L, "excluded"),
        (p_L, p_ML, "ramp up"),
        (p_ML, p_MH, "plateau"),
        (p_MH, p_H, "ramp down"),
        (p_H, 1.0, "floor"),
    ]
    for lo, hi, label in regimes:
        ax.text((lo + hi) / 2, 1.05, label, ha="center", va="bottom", fontsize=8, color="#555")

    fig.tight_layout()
    out = FIGDIR / "fig_g_function.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_frontier(data: dict) -> Path:
    entries = data["frontier"]
    short_labels = {
        "No reg (g=1)": r"$g\equiv 1$",
        "Wide (0.03/0.50/0.70)": "Wide",
        "Default (0.05/0.40/0.60)": "Default",
        "Narrow (0.10/0.30/0.50)": "Narrow",
        "Aggressive (0.15/0.35/0.50/phi=0.05)": "Aggressive",
        "Calibrated": "Calibrated",
    }

    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    seen = set()
    for e in entries:
        x = e["risk_mean"]
        y = e["welfare_mean"]
        label = short_labels.get(e["config"], e["config"])
        key = (round(x, 4), round(y, 0))
        if key in seen:
            continue
        seen.add(key)
        is_baseline = label in (r"$g\equiv 1$", "Calibrated")
        ax.scatter(
            x,
            y,
            s=80 if is_baseline else 60,
            color="#b34" if is_baseline else "#1f3b5c",
            zorder=3,
            edgecolor="white",
            linewidth=0.8,
        )
        dx = 0.0015
        dy = 250
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x + dx, y + dy),
            fontsize=9,
            color="#222",
        )

    ax.set_xlabel("risk exposure $\\rho(\\pi)$ (mean churn among treated)")
    ax.set_ylabel(r"welfare $W(\pi;\mathbf{1})$")
    ax.set_title("Value-conservatism frontier: each non-trivial $g$ costs welfare", pad=10)

    fig.tight_layout()
    out = FIGDIR / "fig_frontier.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_welfare_bars(data: dict) -> Path:
    order = [
        "Random",
        "CATE Only",
        "CATE x LTV",
        "Full Framework (Est.)",
        "Full Framework (Cal.)",
        "S-Learner",
        "Full Framework (Oracle)",
        "tauV Oracle",
    ]
    display = {
        "Random": "Random",
        "CATE Only": "CATE only",
        "CATE x LTV": r"CATE $\times$ LTV",
        "Full Framework (Est.)": "Full (default $g$)",
        "Full Framework (Cal.)": "Full (calibrated $g$)",
        "S-Learner": "S-Learner",
        "Full Framework (Oracle)": "Full-framework oracle ($\\theta_0$)",
        "tauV Oracle": r"$\tau V$ oracle ($g\equiv 1$)",
    }

    welfares = [data["main_results"][k]["welfare_mean"] for k in order]
    sds = [data["main_results"][k]["welfare_sd"] for k in order]
    labels = [display[k] for k in order]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(order))
    colors = []
    for k in order:
        if k == "tauV Oracle":
            colors.append("#1a5a1a")
        elif k == "Full Framework (Oracle)":
            colors.append("#2a7a2a")
        elif k == "S-Learner":
            colors.append("#b34")
        elif k == "Random":
            colors.append("#888")
        else:
            colors.append("#1f3b5c")

    ax.barh(y, welfares, xerr=sds, color=colors, ecolor="#444", capsize=3, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"welfare $W(\pi;\mathbf{1})$ (mean $\pm$ SD across 200 replications)")
    ax.set_title(r"Welfare by strategy ($\kappa=0.30$)", pad=10)

    for yi, w in zip(y, welfares):
        ax.text(w + 400, yi, f"{w:,.0f}", va="center", fontsize=9)

    ax.set_xlim(0, max(welfares) * 1.18)
    fig.tight_layout()
    out = FIGDIR / "fig_welfare_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    _style()
    with DATA.open() as fh:
        data = json.load(fh)
    paths = [fig_g_function(), fig_frontier(data), fig_welfare_bars(data)]
    for p in paths:
        print(f"wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
