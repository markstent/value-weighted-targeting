"""Generate the manuscript figures from the study outputs.

Reads outputs/main.json (fig_frontier, fig_welfare_bars) and outputs/harm_sweep.json
(fig_harm_frontier); fig_g_function is analytic.  Writes vector PDFs to docs/figures/.
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

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "outputs"
FIGDIR = REPO / "docs" / "figures"
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


def fig_g_function(figdir: Path = FIGDIR) -> Path:
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
    out = figdir / "fig_frontier.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_welfare_bars(data: dict, figdir: Path = FIGDIR) -> Path:
    order = [
        "Random",
        "CATE Only",
        "CATE x LTV",
        "Full Framework (Est.)",
        "Full Framework (Cal.)",
        "S-Learner",
        "Oracle hindsight (theta0)",
        "Oracle feasible (g=1)",
        "Oracle hindsight (g=1)",
    ]
    display = {
        "Random": "Random",
        "CATE Only": "CATE only",
        "CATE x LTV": r"CATE $\times$ LTV",
        "Full Framework (Est.)": "Full (default $g$)",
        "Full Framework (Cal.)": "Full (calibrated $g$)",
        "S-Learner": "S-Learner",
        "Oracle hindsight (theta0)": "Hindsight oracle ($\\theta_0$)",
        "Oracle feasible (g=1)": r"Feasible $E[\tau|X]E[V|X]$ oracle ($g\equiv 1$)",
        "Oracle hindsight (g=1)": r"Hindsight $\tau V$ oracle ($g\equiv 1$)",
    }

    welfares = [data["main_results"][k]["welfare_mean"] for k in order]
    sds = [data["main_results"][k]["welfare_sd"] for k in order]
    labels = [display[k] for k in order]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(order))
    colors = []
    for k in order:
        if k == "Oracle hindsight (g=1)":
            colors.append("#1a5a1a")
        elif k == "Oracle feasible (g=1)":
            colors.append("#c58b1d")
        elif k == "Oracle hindsight (theta0)":
            colors.append("#2a7a2a")
        elif k == "S-Learner":
            colors.append("#b34")
        elif k == "Random":
            colors.append("#888")
        else:
            colors.append("#1f3b5c")

    ax.barh(y, welfares, color=colors, alpha=0.9)
    # a missing SD (fewer than 2 replications) gets no error bar rather than a zero-width one
    have = [i for i, sd in enumerate(sds) if sd is not None]
    ax.errorbar([welfares[i] for i in have], y[have], xerr=[sds[i] for i in have],
                fmt="none", ecolor="#444", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"welfare $W(\pi;\mathbf{1})$ (mean $\pm$ SD across replications)")
    ax.set_title(r"Welfare by strategy ($\kappa=0.30$)", pad=10)

    # Place each value label past the end of its error bar (w + sd), not just
    # past the bar end, so the whisker/cap does not overprint the leading digits.
    sd_or_0 = [sd or 0.0 for sd in sds]
    for yi, w, sd in zip(y, welfares, sd_or_0):
        ax.text(w + sd + 700, yi, f"{w:,.0f}", va="center", fontsize=9)

    ax.set_xlim(0, max(w + sd for w, sd in zip(welfares, sd_or_0)) * 1.16)
    fig.tight_layout()
    out = figdir / "fig_welfare_bars.pdf"
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
    cells = [c for c in harm["cells"]
             if c["kappa"] == HARM_KAPPA and c["c_harm"] == HARM_C]
    sigmas = sorted({c["sigma_h"] for c in cells})
    styles = {"tauV": ("#1f3b5c", "o", r"$\tau V$ (harm-blind)"),
              "soft gate": ("#b34", "s", "soft gate"),
              "hard cap": ("#2a7a2a", "^", "hard cap")}
    fig, axes = plt.subplots(1, len(sigmas), figsize=(3.0 * len(sigmas) + 0.6, 3.6),
                             sharey=True, squeeze=False)
    for ax, sigma in zip(axes[0], sigmas):
        for rule, (color, marker, _) in styles.items():
            for oracle in (False, True):
                name = rule + (ORACLE_SUFFIX if oracle else "")
                match = [c for c in cells if c["sigma_h"] == sigma and c["rule"] == name]
                if not match:
                    raise ValueError(
                        f"harm_sweep: no cell for rule {name!r} at sigma_h={sigma:g}, "
                        f"kappa={HARM_KAPPA}, c_harm={HARM_C}")
                c = match[0]
                xe, ye = _ci_err(c["harm_exposure"]), _ci_err(c["welfare"])
                ax.errorbar(
                    c["harm_exposure"]["mean"], c["welfare"]["mean"],
                    xerr=[[xe[0]], [xe[1]]], yerr=[[ye[0]], [ye[1]]],
                    fmt=marker, color=color, ecolor=color, capsize=2, markersize=7,
                    markerfacecolor="white" if oracle else color, markeredgewidth=1.4,
                    elinewidth=1.0, zorder=3)
        ax.set_title(rf"$\sigma_h={sigma:g}$" + (" (oracle harm)" if sigma == 0 else ""))
        ax.set_xlabel("harm exposure among treated")
    axes[0][0].set_ylabel(r"welfare $W(\pi;\mathbf{1})$")
    handles = [plt.Line2D([], [], marker=m, color=col, linestyle="", markersize=7, label=lab)
               for col, m, lab in styles.values()]
    handles += [plt.Line2D([], [], marker="o", color="#555", linestyle="", markersize=7,
                           label="estimated $\\hat\\tau,\\hat V$"),
                plt.Line2D([], [], marker="o", color="#555", linestyle="", markersize=7,
                           markerfacecolor="white", markeredgewidth=1.4,
                           label=r"oracle $\tau,V$ (feasible)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(rf"Welfare vs harm exposure ($\kappa={HARM_KAPPA:.2f}$, $c_{{harm}}={HARM_C:g}$)")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    out = figdir / "fig_harm_frontier.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


FIGURE_NAMES = ("fig_g_function.pdf", "fig_frontier.pdf", "fig_welfare_bars.pdf",
                "fig_harm_frontier.pdf")


def build_all(outdir: Path = OUTDIR, figdir: Path = FIGDIR, smoke: bool = False) -> list[Path]:
    """Render every figure from outdir's JSONs into figdir."""
    _style()
    figdir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    with (outdir / f"main{suffix}.json").open() as fh:
        main = json.load(fh)
    with (outdir / f"harm_sweep{suffix}.json").open() as fh:
        harm = json.load(fh)
    return [fig_g_function(figdir), fig_frontier(main, figdir),
            fig_welfare_bars(main, figdir), fig_harm_frontier(harm, figdir)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--smoke", action="store_true",
                        help="read *_smoke.json and write to outputs/figures_smoke/")
    args = parser.parse_args()
    figdir = SMOKE_FIGDIR if args.smoke else FIGDIR
    for p in build_all(OUTDIR, figdir, args.smoke):
        print(f"wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
