# Value-Weighted Retention Targeting Can Concentrate Harm: Diagnostics and Safeguards for Online Gambling

[![Paper (PDF)](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](docs/stent-2026-value-weighted-targeting.pdf)
[![Repo](https://img.shields.io/badge/GitHub-value--weighted--targeting-181717.svg?logo=github)](https://github.com/markstent/value-weighted-targeting)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-f89939.svg)](https://scikit-learn.org/)
[![Results](https://img.shields.io/badge/results-deterministic%20%26%20reproducible-2ea44f.svg)](#reproducing-the-results)
[![Build](https://img.shields.io/badge/build-LaTeX%20%2F%20tectonic-008080.svg)](#building-the-pdf)
[![License](https://img.shields.io/badge/license-MIT%20(code)%20%2F%20CC%20BY%204.0%20(paper)-blue.svg)](LICENSE)

Companion repository for the working paper *Value-Weighted Retention Targeting Can Concentrate Harm: Diagnostics and Safeguards for Online Gambling* by Mark Stent.

- **Repository:** <https://github.com/markstent/value-weighted-targeting>
- **Read the paper:** [docs/stent-2026-value-weighted-targeting.pdf](docs/stent-2026-value-weighted-targeting.pdf)
- The LaTeX source, figures and generated tables are private and are not part of this repository.

```bash
git clone https://github.com/markstent/value-weighted-targeting.git
```

---

## Abstract

Retention incentives in online gambling are often allocated by ranking players on estimated treatment effect times lifetime value.
Under a fixed budget this value-weighted rule maximises operator value, but operator value and player harm can rise together.
We show that the harm exposure of any budget-exhausting rule exceeds the population mean by the covariance between treatment and harm divided by the budget, and give sufficient conditions for value-weighted ranking to make it non-negative.
In a simulation where harm rises with value, the estimated value-weighted rule raises mean harm among treated players from 0.421 to 0.476 at a 30% budget and leaves a gap of +0.152 at 10%.
Harm tracking value is not enough: in one design an accurate value-weighted rule falls below the population mean despite a strong harm-value correlation.
A harm weight on the score lowers exposure to 0.414, close to what effect-only ranking reaches with no harm input.
With estimated components it raises operator value, but not through harm information: the gain appears when harm falls with value, shrinks or vanishes when harm is independent of value and becomes a cost with accurate components, consistent with trimming the noisy upper tail of the product score.
A hard cap beats the harm weight only with accurate components, and its ceiling holds only for the measure it is given.
In this design, with estimated effects, ranking on the effect alone beats value weighting for every learner at budgets up to 30%; removing effect-churn moderation reverses this.
All parameters are illustrative.

## Key Findings

| Finding | Detail |
|---|---|
| Harm exposure obeys an identity, with sufficient conditions for its sign | For any budget-exhausting rule, exposure of the treated minus the population mean equals the covariance of treatment and harm divided by the budget; value-weighted ranking makes it non-negative when harm rises with value among players with the same treatment effect and ranking on the effect does not steer away from harm, conditions that are sufficient and not necessary |
| Value-weighted ranking concentrates harm in the main design, most at tight budgets | Mean harm among treated players is 0.476 against a population mean of 0.421, a gap of +0.055 (95% CI +0.051 to +0.058) at kappa = 0.30 and +0.152 (95% CI +0.146 to +0.158) at kappa = 0.10; the feasible oracle gap is +0.033 (+0.032 to +0.034); the gap follows the harm-value association in the main design, but with feasible-oracle components effect-only ranking has a gap of -0.046 and in a decoupled design the feasible-oracle value-weighted rule has a gap of -0.014 although corr(harm, value) is +0.82 |
| A soft harm weight lowers exposure, but its value gain is not a harm effect | Exposure falls to 0.414 at kappa = 0.30 (gap +0.018 at kappa = 0.10), close to the 0.416 that effect-only ranking reaches with no harm input; with estimated components operator value rises from 16,446 to 19,045 (+2,598, 95% CI 2,406 to 2,790), but the gain also appears when harm falls with value (+468, 95% CI 419 to 517), vanishes at kappa = 0.30 when harm is independent of value (-60, 95% CI -135 to 16), where at kappa = 0.10 random demotion still helps (+312, 95% CI 255 to 369) and becomes a cost with feasible-oracle components (2.5%), consistent with trimming the noisy upper tail of the product score where effects are small or negative; choose it for harm reduction and do not count on the gain transferring |
| A hard harm cap needs accurate inputs | With estimated components the cap earns 15,701 at exposure 0.413, which is 3,344 less than the soft weight (95% CI -3,539 to -3,148); it is cheaper only with feasible-oracle components (+376, 95% CI 326 to 426), and with a noisy harm measure true exposure overshoots the ceiling by +0.013 (+0.012 to +0.014) at sigma_h = 0.5 and +0.043 (+0.041 to +0.044) at sigma_h = 1 |
| In the main design, value weighting pays only with accurate components | With feasible-oracle effects and values it beats effect-only ranking at every budget (+1,441 at kappa = 0.10, 95% CI 1,383 to 1,499); with estimated effects, effect-only ranking wins for every learner at budgets up to kappa = 0.30 (T-learner, proxy value model: -1,482 for value weighting at kappa = 0.10, 95% CI -1,632 to -1,332, and -187 at 0.30, 95% CI -319 to -55; -385 on clip-adjusted welfare) and in 13 of 16 learner cells at 0.50; the null scenario without effect-churn moderation reverses this (+5,157, 95% CI 4,704 to 5,610), as do the misspecified-effect (+287) and asymmetric (+398) scenarios |
| The crossover budget belongs to the learner | With the T-learner and the proxy value model kappa* is 0.350 (95% CI 0.325 to 0.375) and the X-learner gives 0.425; with the S-learner value weighting never crosses within the grid; with feasible-oracle components there is no crossing |
| Estimation cost is measured against a feasible oracle | A covariate-only oracle reaches 73.5% (95% CI 73.3% to 73.7%) of the hindsight oracle; a Shapley split attributes 99.9% to 101.1% of regret to the effect estimate |
| Preference-weight calibration does not help | Plug-in calibration selects the identity in 200 of 200 replications and AIPW calibration costs -881 (95% CI -1,217 to -544) against the identity |

All parameters are illustrative; there is no operator data.

## Repository Structure

```
LICENSE
pytest.ini
requirements.txt
requirements-dev.txt
config/
  tuned_params.json               Frozen learner tuning used by the studies
docs/
  stent-2026-value-weighted-targeting.pdf   Compiled manuscript
paper/                            Private LaTeX source, figures/ and generated/ (numbers.tex, tab_*.tex); not in the public repository, git-ignored
src/
  core/                           Shared data generation, estimators, oracle, harm and statistics code
  study_main.py                   Main study: strategies, oracles, budget sweep, calibration
  study_harm.py                   Harm sweep: soft weight vs hard cap, harm-value alignment, harm-measure noise
  study_budget.py                 Budget sweep (effect source x value source) and crossover budget
  study_robustness.py             Robustness scenarios and calibration checks
  study_learners.py               Learner (T/X/S/DR) x value model comparison with frozen tuning and Shapley regret split
  run_all.py                      One-command pipeline (studies, manifest, figures, generated TeX)
  make_figures.py                 Figure generation (PDFs in paper/figures/)
  build_paper.py                  Compiles the private paper/ LaTeX source and copies the PDF to docs/
  make_tex.py                     Writes paper/generated/numbers.tex and the tables from outputs/*.json
tests/                            pytest suite, including golden-file and pipeline tests
outputs/                          Simulation results (JSON) and run_manifest.json; regenerated by the scripts
results/                          Tracked copy of the final full run: the five result JSONs and run_manifest.json
```

## Reproducing the Results

All quantities in the paper are produced by simulation; there is no empirical data.
The study is **deterministic**: every replication `r` uses a fixed seed derived from `r`, so the scripts reproduce the reported tables and figures exactly.
The paper reads every number from `paper/generated/numbers.tex`, so no figure in the text is typed by hand.
A successful full run of `python src/run_all.py` on a clean git tree also publishes the five result JSONs and `run_manifest.json` to the tracked `results/` folder.
The repository tracks the final run's `results/`, and `python src/run_all.py` reproduces it.

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
python src/make_figures.py                  # paper/figures/*.pdf
python src/make_tex.py                      # paper/generated/numbers.tex and tab_*.tex
```

The manifest records the git SHA and dirty flag, package versions, run-size arguments, per-step wall time and exit code, the sha256 of every output JSON and figure, and the sha256 of `config/tuned_params.json`.
Tests: `pytest` runs everything (manuscript checks that need `paper/` are skipped in a fresh clone) including `tests/test_pipeline.py::test_smoke_fresh_clone` (marked `slow`; it clones the repo and runs the smoke pipeline).
Use `pytest -m "not slow"` to skip it and `pytest -m slow` to run only it.

## Building the PDF

The LaTeX source lives in the private `paper/` folder, so a fresh clone cannot build the PDF.
The compiled paper is tracked in `docs/stent-2026-value-weighted-targeting.pdf`.
With `paper/` present, `python src/build_paper.py` compiles `paper/stent-2026-value-weighted-targeting.tex` and copies the PDF to `docs/`.
It uses [tectonic](https://tectonic-typesetting.github.io/) if installed and `latexmk` otherwise.
It fails on undefined references, citations or macros.
Without `paper/` it exits with a message saying the source is private.
The bibliography is inline (`thebibliography`), so no separate BibTeX run is needed.

```bash
python src/build_paper.py
```

## Citation

```bibtex
@unpublished{stent2026bonus,
  title  = {Value-Weighted Retention Targeting Can Concentrate Harm: Diagnostics and Safeguards for Online Gambling},
  author = {Stent, Mark},
  year   = {2026},
  note   = {Working paper},
  url    = {https://github.com/markstent/value-weighted-targeting}
}
```

## Responsible Use

The paper studies promotional targeting in online gambling, a harm-sensitive domain.
It shows that value-weighted ranking can concentrate incentives on harmed players, and it treats an exclusion set, a harm-aware score and a fail-closed default as preconditions for any deployment.
With estimated components the paper prefers the soft harm weight over the hard cap for harm reduction, and, where a guarantee is required, an audited harm measure and an exclusion set, because a cap binds only on the harm measure it is given.
The soft weight's welfare gain in the simulation is not attributable to harm information, so it should not be expected to transfer.
The paper distinguishes *operator welfare* (the quantity optimised) from player wellbeing.
It should not be deployed in a gambling setting without those safeguards and the validated harm-measurement and regulatory infrastructure they depend on.
See the paper's Responsible Deployment and Ethics sections.

## License

MIT (code) / CC BY 4.0 (paper). See [LICENSE](LICENSE).
