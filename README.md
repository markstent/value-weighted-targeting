# When Does Value-Weighted Targeting Pay (and When It Does Not)?

### A Diagnostic for Budget-Constrained Retention Targeting in Online Gaming

[![Paper (PDF)](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper.pdf)
[![Repo](https://img.shields.io/badge/GitHub-bonus--allocation--framework-181717.svg?logo=github)](https://github.com/markstent/bonus-allocation-framework)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-f89939.svg)](https://scikit-learn.org/)
[![Results](https://img.shields.io/badge/results-deterministic%20%26%20reproducible-2ea44f.svg)](#reproducing-the-results)
[![Build](https://img.shields.io/badge/build-LaTeX%20%2F%20tectonic-008080.svg)](#building-the-pdf)
[![License](https://img.shields.io/badge/license-all%20rights%20reserved-lightgrey.svg)](LICENSE)

Companion repository for the working paper *When Does Value-Weighted Targeting Pay (and When It Does Not)? A Diagnostic for Budget-Constrained Retention Targeting in Online Gaming* by Mark Stent.

- **Repository:** <https://github.com/markstent/bonus-allocation-framework>
- **Read the paper:** [paper.pdf](paper.pdf) &nbsp;·&nbsp; latest source: [`docs/stent-2026-value-weighted-targeting.tex`](docs/stent-2026-value-weighted-targeting.tex)

```bash
git clone https://github.com/markstent/bonus-allocation-framework.git
```

---

## Abstract

Budget-constrained treatment targeting -- selecting the top *k* individuals for intervention from a heterogeneous population -- is a central problem in personalised marketing, healthcare, and public policy. Practitioners routinely rank candidates by treatment effect times outcome value; this paper places that heuristic inside budget-constrained empirical welfare maximisation, where it is welfare-optimal as a corollary of the standard threshold result applied to a value-weighted outcome, and extends it with an operator-configurable preference weight. The answer the paper reaches is largely cautionary. Value weighting pays only at generous budgets: at a realistic campaign budget (κ ≈ 0.10) plain treatment-effect targeting matches or beats it, and a hybrid joint model (a joint treatment-effect model with a separate value model) outperforms the decomposed product rule at every budget tested. The operator-preference weight provably cannot raise welfare, so it prices restrictions rather than discovering them. The sharpest, most domain-specific finding is that value weighting is **harm-concentrating by construction**: because a gambler's lifetime value is high precisely when they are losing the most, the welfare optimum directs treatment at the most-harmed players and is unsafe by default. The paper's remedy is to internalise harm into the objective (a harm-aware score, or a hard harm-exposure constraint) rather than bolt it on, and to price the welfare cost of that safety. "Welfare" throughout denotes operator welfare, distinct from player wellbeing. The reported numbers are illustrative rather than transportable; the durable contribution is the procedure and the diagnostic, not the point estimates.

## Key Findings

| Finding | Detail |
|---|---|
| Value-weighting is harm-concentrating by construction | Where value correlates with harm (gambling LTV), the welfare optimum directs treatment at the most-harmed players (harm exposure 0.466 vs 0.42 population mean); the harm-blind rule is unsafe by default, and the fix is to internalise harm into the objective, not filter after |
| Value-weighting is budget-contingent, with an estimated crossover | Pure CATE beats value weighting at tight budgets and the ranking reverses as the budget grows; the crossover budget is κ\* ≈ 0.34 (symmetric moderation) / 0.25 (asymmetric), with bootstrap CIs |
| Value-weighting helps under misspecification *only when V carries the dropped signal* | An adversarial misspecification whose dropped driver is orthogonal to value removes the benefit and, at κ = 0.10, reverses it (−25%) |
| The operator-preference weight cannot improve welfare | Calibration returns the neutral setting (Proposition 3); it holds under a doubly-robust (AIPW) objective too, where identity-selection drops to 14/200 but true welfare is not improved |
| The hybrid architecture leads on welfare | A joint treatment-effect model with a separate value model beats both the pure decomposition and the pure end-to-end model at every budget |
| "Joint model is more robust" is a mirage | The joint model looks immune to CATE misspecification only because the feature is dropped from one side; drop it from the joint model too and it degrades comparably (64%). The advantage is architectural (cross-component interactions), not robustness to missing inputs |
| The regret decomposition is a working diagnostic | It localises welfare loss to a component and correctly fingers the degraded one - including in an observational (no-oracle) plug-in version |
| Harm protection is quantifiable, and a hard constraint dominates a soft weight | A hard harm constraint reaches a given harm ceiling at lower welfare cost than the soft multiplicative gate, and cannot be overridden by a large τ·V |

## Repository Structure

```
paper.pdf                         The paper (stable link)
LICENSE
requirements.txt
docs/
  stent-2026-value-weighted-targeting.tex   LaTeX source (current version)
  stent-2026-value-weighted-targeting.pdf   Compiled manuscript
  figures/                        Figures used in the manuscript
  review_v39/                     Simulated peer-review package that drove the revisions
src/
  simulation_v38.py               Core simulation (all scenarios, calibration, robustness)
  study_robustness.py             Robustness scenarios: null/correlated/asymmetric/heavy-tail DGPs,
                                  corr(tau,V) sweep, clip-adjusted welfare, component update,
                                  plug-in diagnostic, AIPW calibration, adversarial and
                                  like-for-like misspecification
  study_learners.py               CATE learner (T/X/S/DR/R) x value model comparison with frozen
                                  tuning; order-free regret split; diagnostic; E2E model
  compute_tauv_oracle*.py         tau*V oracle welfare benchmarks
  figures_v39.py                  Figure generation
outputs/                          Simulation results (JSON); regenerated by the scripts
```

## Reproducing the Results

All quantities in the paper are produced by simulation; there is no empirical data. The study is **deterministic**: every replication `r` uses the fixed seed `42 + 7r`, so the scripts reproduce the reported tables and figures exactly.

```bash
# Python 3.12
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# core study (main results, calibration, budget sweep, frontier, robustness)
python src/simulation_v38.py

# revision add-ons (each writes a JSON to outputs/)
python src/study_robustness.py              # all robustness scenarios (robustness.json)
python src/study_learners.py                # CATE learner comparison, tuning, regret split, diagnostic
```

## Building the PDF

Requires [tectonic](https://tectonic-typesetting.github.io/) or a standard LaTeX distribution (`pdflatex`). The bibliography is inline (`thebibliography`), so no separate BibTeX run is needed.

```bash
cd docs && tectonic stent-2026-value-weighted-targeting.tex
```

## Citation

```bibtex
@unpublished{stent2026bonus,
  title  = {When Does Value-Weighted Targeting Pay (and When It Does Not)? A Diagnostic for Budget-Constrained Retention Targeting in Online Gaming},
  author = {Stent, Mark},
  year   = {2026},
  note   = {Working paper},
  url    = {https://github.com/markstent/bonus-allocation-framework}
}
```

## Responsible Use

The paper studies promotional targeting in online gambling, a harm-sensitive domain. The framework is presented with explicit responsible-gaming safeguards (a vulnerability-based exclusion set and a harm gate treated as deployment preconditions, conservative defaults, and a hard harm-exposure constraint) and draws a clear distinction between *operator welfare* (the econometric quantity optimised) and *player welfare*. It should not be deployed in a gambling setting without those safeguards and the validated harm-measurement and regulatory infrastructure they depend on. See the paper's Responsible-Gaming and Ethics sections.

## License

All rights reserved; provided for review, reference, and reproducibility. See [LICENSE](LICENSE).
