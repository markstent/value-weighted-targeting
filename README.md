# Value-Weighted Retention Targeting Concentrates Harm

### A Diagnostic and a Harm-Aware Rule for Budget-Constrained Targeting in Online Gambling

[![Paper (PDF)](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper.pdf)
[![Repo](https://img.shields.io/badge/GitHub-bonus--allocation--framework-181717.svg?logo=github)](https://github.com/markstent/bonus-allocation-framework)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-f89939.svg)](https://scikit-learn.org/)
[![Results](https://img.shields.io/badge/results-deterministic%20%26%20reproducible-2ea44f.svg)](#reproducing-the-results)
[![Build](https://img.shields.io/badge/build-LaTeX%20%2F%20tectonic-008080.svg)](#building-the-pdf)
[![License](https://img.shields.io/badge/license-all%20rights%20reserved-lightgrey.svg)](LICENSE)

Companion repository for the working paper *Value-Weighted Retention Targeting Concentrates Harm: A Diagnostic and a Harm-Aware Rule for Budget-Constrained Targeting in Online Gambling* by Mark Stent.

- **Repository:** <https://github.com/markstent/bonus-allocation-framework>
- **Read the paper:** [paper.pdf](paper.pdf) &nbsp;·&nbsp; latest source: [`docs/stent-2026-value-weighted-targeting.tex`](docs/stent-2026-value-weighted-targeting.tex)

```bash
git clone https://github.com/markstent/bonus-allocation-framework.git
```

---

## Abstract

Retention incentives in online gambling are often allocated by ranking players on estimated treatment effect times lifetime value.
Under a fixed budget this value-weighted rule maximises operator value, but operator value and player harm can rise together.
We show that the harm exposure of any budget-exhausting rule exceeds the population mean by the covariance between treatment and harm divided by the budget, and give a sufficient condition under which value-weighted ranking makes that covariance positive: harm rises with value among players with the same treatment effect.
In a simulation with 200 replications of 10,000 players, value-weighted ranking raises mean harm among treated players from 0.421 to 0.476, and the sign of the gap follows the sign of the harm-value association.
A harm-aware rule that multiplies the score by a harm weight brings exposure to 0.414 and, with estimated components, raises operator value from 16,446 to 19,045.
A hard harm cap is optimal for a given ceiling in the population, but it beats the harm weight only with feasible-oracle components, and its ceiling holds only for the harm measure it is given.
With estimated treatment effects, ranking on the effect alone yields more operator value than value weighting for every learner we test at tight and moderate budgets, so the budget at which value weighting starts to pay depends on the learner.
A feasible oracle that uses only covariates reaches 73.5% of the hindsight oracle, which is the right benchmark for estimation cost.
All parameters are illustrative.

## Key Findings

| Finding | Detail |
|---|---|
| Harm exposure obeys an exact identity | For any budget-exhausting rule, exposure of the treated minus the population mean equals the covariance of treatment and harm divided by the budget; value-weighted ranking makes it positive when harm rises with value among players with the same treatment effect |
| Value-weighted ranking concentrates harm in simulation | Mean harm among treated players is 0.476 against a population mean of 0.421, a gap of +0.055 (95% CI +0.051 to +0.058); the feasible oracle gap is +0.033 (+0.032 to +0.034); the sign of the gap follows the harm-value association (sample correlation from -0.75 to +0.80 across the sweep) |
| A soft harm weight removes the concentration at no operator-value cost | Exposure falls to 0.414, below the population mean, and operator value rises from 16,446 to 19,045, a difference of +2,598 (95% CI 2,406 to 2,790, Cohen's d 1.89); with estimated components and harm rising with value |
| A hard harm cap needs accurate inputs | With estimated components the cap earns 15,701 at exposure 0.413, which is 3,344 less than the soft weight (95% CI -3,539 to -3,148); it is cheaper only with feasible-oracle components (+378, 95% CI 330 to 426), and with a noisy harm measure true exposure overshoots the ceiling by +0.013 (+0.012 to +0.014) at sigma_h = 0.5 and +0.043 (+0.041 to +0.044) at sigma_h = 1 |
| Value weighting is an oracle property | With feasible-oracle effects and values it beats effect-only ranking at every budget (+1,445 at kappa = 0.10, 95% CI 1,388 to 1,503); with estimated effects, effect-only ranking wins for every learner at budgets up to kappa = 0.30 (T-learner, proxy value model: -1,482 for value weighting at kappa = 0.10, 95% CI -1,632 to -1,332, and -187 at 0.30, 95% CI -319 to -55) |
| The crossover budget belongs to the learner | With the T-learner and the proxy value model kappa* is 0.350 (95% CI 0.325 to 0.375) and the X-learner gives 0.425; with the S-learner value weighting never crosses within the grid; with feasible-oracle components there is no crossing |
| Estimation cost is measured against a feasible oracle | A covariate-only oracle reaches 73.5% (95% CI 73.3% to 73.7%) of the hindsight oracle; a Shapley split attributes 99.9% to 101.2% of regret to the effect estimate |
| Preference-weight calibration does not help | Plug-in calibration selects the identity in 200 of 200 replications and AIPW calibration costs -1,437 (95% CI -1,828 to -1,046) against the identity |

All parameters are illustrative; there is no operator data.

## Repository Structure

```
paper.pdf                         The paper (stable link)
LICENSE
pytest.ini
requirements.txt
requirements-dev.txt
config/
  tuned_params.json               Frozen learner tuning used by the studies
docs/
  stent-2026-value-weighted-targeting.tex   LaTeX source (current version)
  stent-2026-value-weighted-targeting.pdf   Compiled manuscript
  figures/                        Figures used in the manuscript
  generated/                      numbers.tex (every number in the text) and tab_*.tex tables, written by make_tex.py
  peer_review/                    Simulated peer-review package that drove the revisions
src/
  core/                           Shared data generation, estimators, oracle, harm and statistics code
  study_main.py                   Main study: strategies, oracles, budget sweep, calibration
  study_harm.py                   Harm sweep: soft weight vs hard cap, harm-value alignment, harm-measure noise
  study_budget.py                 Budget sweep (effect source x value source) and crossover budget
  study_robustness.py             Robustness scenarios and calibration checks
  study_learners.py               Learner (T/X/S/DR) x value model comparison with frozen tuning and Shapley regret split
  run_all.py                      One-command pipeline (studies, manifest, figures, generated TeX)
  make_figures.py                 Figure generation (PDFs in docs/figures/)
  make_tex.py                     Writes docs/generated/numbers.tex and the tables from outputs/*.json
tests/                            pytest suite, including golden-file and pipeline tests
outputs/                          Simulation results (JSON) and run_manifest.json; regenerated by the scripts
```

## Reproducing the Results

All quantities in the paper are produced by simulation; there is no empirical data.
The study is **deterministic**: every replication `r` uses a fixed seed derived from `r`, so the scripts reproduce the reported tables and figures exactly.
The paper reads every number from `docs/generated/numbers.tex`, so no figure in the text is typed by hand.

```bash
# Python 3.12
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# everything: all five studies, then figures, generated TeX, and outputs/run_manifest.json
python src/run_all.py                       # full size (N=10000, 200 reps); --jobs N runs N studies in parallel
python src/run_all.py --smoke --jobs 4      # quick check (3 reps, n=2000); outputs go to smoke paths
python src/run_all.py --only main,harm      # a subset of studies

# or one step at a time (the studies accept --smoke, --reps, --n-players)
python src/study_main.py                    # main.json
python src/study_harm.py                    # harm_sweep.json
python src/study_budget.py                  # budget_2x2.json
python src/study_learners.py                # learners.json
python src/study_robustness.py              # robustness.json
python src/make_figures.py                  # docs/figures/*.pdf
python src/make_tex.py                      # docs/generated/numbers.tex and tab_*.tex
```

The manifest records the git SHA and dirty flag, package versions, run-size arguments, per-step wall time and exit code, the sha256 of every output JSON and figure, and the sha256 of `config/tuned_params.json`.
Tests: `pytest` runs everything including `tests/test_pipeline.py::test_smoke_fresh_clone` (marked `slow`; it clones the repo and runs the smoke pipeline).
Use `pytest -m "not slow"` to skip it and `pytest -m slow` to run only it.

## Building the PDF

Requires [tectonic](https://tectonic-typesetting.github.io/) or a standard LaTeX distribution (`pdflatex`). The bibliography is inline (`thebibliography`), so no separate BibTeX run is needed.

```bash
cd docs && tectonic stent-2026-value-weighted-targeting.tex
```

## Citation

```bibtex
@unpublished{stent2026bonus,
  title  = {Value-Weighted Retention Targeting Concentrates Harm: A Diagnostic and a Harm-Aware Rule for Budget-Constrained Targeting in Online Gambling},
  author = {Stent, Mark},
  year   = {2026},
  note   = {Working paper},
  url    = {https://github.com/markstent/bonus-allocation-framework}
}
```

## Responsible Use

The paper studies promotional targeting in online gambling, a harm-sensitive domain.
It shows that value-weighted ranking can concentrate incentives on harmed players, and it treats an exclusion set, a harm-aware score and a fail-closed default as preconditions for any deployment.
With estimated components the paper recommends the soft harm weight over the hard cap, and, where a guarantee is required, an audited harm measure and an exclusion set, because a cap binds only on the harm measure it is given.
The paper distinguishes *operator welfare* (the quantity optimised) from player wellbeing.
It should not be deployed in a gambling setting without those safeguards and the validated harm-measurement and regulatory infrastructure they depend on.
See the paper's Responsible Deployment and Ethics sections.

## License

All rights reserved; provided for review, reference, and reproducibility. See [LICENSE](LICENSE).
