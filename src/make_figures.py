"""Generate the manuscript figures from the study outputs.

Reads outputs/main.json (fig_frontier) and outputs/harm_sweep.json
(fig_harm_frontier); fig_g_function is analytic.  Writes vector PDFs to paper/figures/.
With --smoke it reads the *_smoke.json outputs and writes to outputs/figures_smoke/
instead, so a smoke run never overwrites the manuscript figures.  Re-run with:

    python src/make_figures.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# Figures are drawn at the width they are printed (the elsarticle preprint text block is 16 cm),
# so the 9-10 pt fonts below appear at roughly the size of the caption text.  Titles are left to
# the LaTeX captions.
TEXT_WIDTH_IN = 16.0 / 2.54
OUTDIR = REPO / "outputs"
FIGDIR = REPO / "paper" / "figures"
SMOKE_FIGDIR = OUTDIR / "figures_smoke"

HARM_KAPPA = 0.30
HARM_C = 1.0
ORACLE_SUFFIX = " (oracle tau,V)"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "mathtext.fontset": "cm",
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


def fig_g_function(figdir: Path = FIGDIR) -> Path:
    theta = (0.05, 0.40, 0.60, 0.90, 0.20)
    p_L, p_ML, p_MH, p_H, phi = theta
    p = np.linspace(0, 1, 1000)
    g = g_piecewise(p, theta)

    fig, ax = plt.subplots(figsize=(0.8 * TEXT_WIDTH_IN, 2.9))
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
    ax.annotate(r"$g_{\min}$", xy=(0.99, phi + 0.03), ha="right", va="bottom", fontsize=9, color="#b34")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.1)
    # Drop the numeric ticks at 0.4 and 0.6: they overprint the p_ML / p_MH
    # breakpoint labels below the axis. The breakpoint symbols mark those
    # positions, and the caption gives the theta values.
    ax.set_xticks([0.0, 0.2, 0.8, 1.0])
    ax.set_xlabel("churn probability $p$")
    ax.set_ylabel(r"risk-modulation weight $g(p;\theta)$")

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
    out = figdir / "fig_g_function.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_frontier(data: dict, figdir: Path = FIGDIR) -> Path:
    entries = data["frontier"]
    short_labels = {
        "No reg (g=1)": r"$g\equiv 1$",
        "Wide (0.03/0.50/0.70)": "Wide",
        "Default (0.05/0.40/0.60)": "Default",
        "Narrow (0.10/0.30/0.50)": "Narrow",
        "Aggressive (0.15/0.35/0.50/phi=0.05)": "Aggressive",
        "Calibrated": "Calibrated",
    }

    fig, ax = plt.subplots(figsize=(0.8 * TEXT_WIDTH_IN, 3.0))

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
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=9,
            color="#222",
        )

    ax.set_xlabel("risk exposure (mean churn probability among treated)")
    ax.set_ylabel(r"welfare $\mathcal{W}(\pi;\mathbf{1})$")
    ax.margins(x=0.08, y=0.1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    out = figdir / "fig_frontier.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _ci_err(entry: dict) -> tuple[float, float]:
    """Asymmetric half-widths (lo, hi) from a {'mean', 'ci95'} entry; zero if no CI."""
    ci = entry.get("ci95")
    if not ci:
        return 0.0, 0.0
    return entry["mean"] - ci[0], ci[1] - entry["mean"]


def fig_harm_frontier(harm: dict, figdir: Path = FIGDIR) -> Path:
    """Welfare vs harm exposure among the treated, one panel per sigma_h.

    kappa = 0.30, c_harm = 1.0.  Filled markers rank on estimated tau_hat * ltv_hat;
    hollow markers on the feasible oracle E[tau|X] E[V|X] ("oracle tau,V").  Error
    bars are the 95% intervals stored in harm_sweep.json.  Welfare and exposure are
    both evaluated on true quantities.
    """
    cells = [c for c in harm["cells"] if c["dgp"] == "main" and c["kappa"] == HARM_KAPPA and c["c_harm"] == HARM_C]
    sigmas = sorted({c["sigma_h"] for c in cells})
    styles = {
        "tauV": ("#1f3b5c", "o", r"$\tau V$ (harm-blind)"),
        "soft gate": ("#b34", "s", "soft weight"),
        "hard cap": ("#2a7a2a", "^", "hard cap"),
    }
    # Shared axes: every panel has the same exposure and welfare ranges, so positions compare across panels.
    fig, axes = plt.subplots(1, len(sigmas), figsize=(TEXT_WIDTH_IN, 3.0), sharex=True, sharey=True, squeeze=False)
    for ax, sigma in zip(axes[0], sigmas):
        for rule, (color, marker, _) in styles.items():
            for oracle in (False, True):
                name = rule + (ORACLE_SUFFIX if oracle else "")
                match = [c for c in cells if c["sigma_h"] == sigma and c["rule"] == name]
                if not match:
                    raise ValueError(
                        f"harm_sweep: no cell for rule {name!r} at sigma_h={sigma:g}, "
                        f"kappa={HARM_KAPPA}, c_harm={HARM_C}"
                    )
                c = match[0]
                xe, ye = _ci_err(c["harm_exposure"]), _ci_err(c["welfare"])
                ax.errorbar(
                    c["harm_exposure"]["mean"],
                    c["welfare"]["mean"],
                    xerr=[[xe[0]], [xe[1]]],
                    yerr=[[ye[0]], [ye[1]]],
                    fmt=marker,
                    color=color,
                    ecolor=color,
                    capsize=2,
                    markersize=7,
                    markerfacecolor="white" if oracle else color,
                    markeredgewidth=1.4,
                    elinewidth=1.0,
                    zorder=3,
                )
        ax.set_title(rf"$\sigma_h={sigma:g}$" + (" (oracle harm)" if sigma == 0 else ""))
        ax.margins(x=0.12)
    axes[0][len(sigmas) // 2].set_xlabel("harm exposure among treated")
    axes[0][0].set_ylabel(r"welfare $\mathcal{W}(\pi;\mathbf{1})$")
    axes[0][0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    handles = [
        plt.Line2D([], [], marker=m, color=col, linestyle="", markersize=7, label=lab)
        for col, m, lab in styles.values()
    ]
    handles += [
        plt.Line2D(
            [], [], marker="o", color="#555", linestyle="", markersize=7, label="estimated $\\hat\\tau,\\hat V$"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            color="#555",
            linestyle="",
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.4,
            label=r"oracle $\tau,V$ (feasible)",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9, handletextpad=0.3, columnspacing=1.2
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    out = figdir / "fig_harm_frontier.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


FIGURE_NAMES = ("fig_g_function.pdf", "fig_frontier.pdf", "fig_harm_frontier.pdf")


def build_all(outdir: Path = OUTDIR, figdir: Path = FIGDIR, smoke: bool = False) -> list[Path]:
    """Render every figure from outdir's JSONs into figdir."""
    _style()
    figdir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    with (outdir / f"main{suffix}.json").open() as fh:
        main = json.load(fh)
    with (outdir / f"harm_sweep{suffix}.json").open() as fh:
        harm = json.load(fh)
    return [
        fig_g_function(figdir),
        fig_frontier(main, figdir),
        fig_harm_frontier(harm, figdir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--smoke", action="store_true", help="read *_smoke.json and write to outputs/figures_smoke/")
    args = parser.parse_args()
    figdir = SMOKE_FIGDIR if args.smoke else FIGDIR
    for p in build_all(OUTDIR, figdir, args.smoke):
        print(f"wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
