# When Does Value-Weighted Targeting Pay? Budget-Constrained Retention Targeting in Online Gaming

Companion repository for the paper *When Does Value-Weighted Targeting Pay? Budget-Constrained Retention Targeting in Online Gaming* by Mark Stent.

**[PDF](docs/manuscript_v41.pdf)**

---

## Abstract

Budget-constrained treatment targeting -- selecting the top *k* individuals for intervention from a heterogeneous population -- is a central problem in personalised marketing, healthcare, and public policy. This paper places the practitioner heuristic of ranking by treatment effect times outcome value inside the empirical-welfare-maximisation framework, showing it is the welfare-maximising budget-constrained policy and extending it with an operator-configurable preference weight. Its more durable contributions are a regret decomposition that identifies which estimated component is costing welfare, a characterisation of when value weighting helps versus hurts (it is budget- and model-quality-contingent), and a method to price any targeting restriction an operator is forced to impose. A simulation study across two structurally different moderation structures, plus a harm-gate analysis and a clip-adjusted welfare check, validates the predictions and quantifies the welfare-interpretability trade-off between decomposed and single-model approaches. "Welfare" throughout denotes operator welfare, distinct from player wellbeing.

## Key Findings

| Finding | Detail |
|---------|--------|
| Value-weighting's advantage is budget- and model-quality-contingent | Helps under CATE misspecification and at generous budgets; pure CATE is competitive or better at realistic tight budgets (kappa = 0.10) |
| Calibration recovers *g* = 1 across all scenarios | Consistent with the value-conservatism trade-off (Proposition 3); calibration prices preferences, it does not discover them |
| Joint modelling (S-Learner) leads on welfare | ~46% above decomposed strategies (depth-matched: +53% to +74%); unaffected by CATE misspecification that costs the decomposition 60% |
| Harm protection has a quantifiable cost | Oracle harm gate: 11.8% reduction in harm exposure at 6.4% operator-welfare cost (Corollary 1 made concrete) |
| Nominal-welfare bias is modest | Clip-adjusted re-evaluation: 3.7%-6.7% overstatement, rankings unchanged |

## Repository Structure

```
docs/
  manuscript_v41.tex            LaTeX source (current version)
  manuscript_v41.pdf            Compiled manuscript
  review_v39/                   Peer-review package on v39 (editorial decision, reviewer reports) that drove the v40 revision
src/
  simulation_v38.py             Core simulation (all scenarios, calibration, robustness)
  compute_depth_ablation.py     Depth-matched S-Learner / T-Learner ablation
  compute_revision_additions.py Revision-40 additions (asymmetric DGP, S-Learner in
                                misspec/null, clip-adjusted welfare, harm gate)
  compute_v41_additions.py      Revision-41 additions (regret-decomposition diagnostic;
                                end-to-end joint value-weighted model)
  compute_tauv_oracle*.py       tau*V oracle welfare benchmarks
  figures_v39.py                Figure generation
outputs/
  simulation_v29.json           Core simulation results (matches manuscript tables)
  simulation_v40_additions.json Revision-40 additions results
  simulation_v41_additions.json Revision-41 additions results
```

## Simulation

`src/simulation_v38.py` implements the core simulation study: a synthetic population of 10,000 individuals with nonlinear CATE, churn moderation, and misspecified LTV, evaluated across seven targeting strategies and three scenarios (well-specified CATE, misspecified CATE, null DGP) over 200 replications, with budget sensitivity, the value-conservatism frontier, and robustness checks (X-Learner, correlated features).

`src/compute_revision_additions.py` runs the revision-40 additions on the same 200 seeds: an asymmetric-moderation second DGP, the S-Learner in the misspecification and null scenarios, a clip-adjusted (effective-tau) welfare re-evaluation, and a harm-gate simulation demonstrating the welfare cost of harm protection (Corollary 1).

`src/compute_v41_additions.py` runs the revision-41 additions: the regret-decomposition diagnostic (showing the telescoping bound correctly localises error to the degraded component) and an end-to-end joint value-weighted model (the un-handicapped joint comparator).

Requires Python 3.10+ with `numpy`, `scipy`, and `scikit-learn`.

```bash
python src/simulation_v38.py
python src/compute_revision_additions.py        # full 200-run revision-40 additions
python src/compute_v41_additions.py             # full 200-run revision-41 additions
python src/compute_depth_ablation.py
```

## Building the PDF

Requires [tectonic](https://tectonic-typesetting.github.io/) or a standard LaTeX distribution (pdflatex).

```bash
cd docs && tectonic manuscript_v41.tex
```

## Citation

```bibtex
@unpublished{stent2026bonus,
  title   = {When Does Value-Weighted Targeting Pay? Budget-Constrained Retention Targeting in Online Gaming},
  author  = {Stent, Mark},
  year    = {2026},
  note    = {Working paper}
}
```

## License

All rights reserved. This repository is provided for review and reference purposes.
