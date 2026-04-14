# Who Gets the Bonus? Welfare-Optimal Treatment Targeting with Value Weights and Risk Modulation

Companion repository for the paper *Who Gets the Bonus? Welfare-Optimal Treatment Targeting with Value Weights and Risk Modulation* by Mark Stent.

**[PDF](docs/manuscript_v38.pdf)**

---

## Abstract

Budget-constrained treatment targeting -- selecting the top *k* individuals for intervention from a heterogeneous population -- is a central problem in personalised marketing, healthcare, and public policy. This paper defines a welfare function that weights each individual's treatment effect by their outcome value and an operator-configurable risk-modulation function, and shows that the welfare-maximising deterministic rule under a budget constraint ranks individuals by a composite score decomposed into three independently estimable components. The value-conservatism trade-off quantifies the welfare cost of the operator's targeting preferences, enabling informed decisions about when risk modulation is worth the cost. A simulation study across three scenarios validates the theoretical predictions and quantifies the welfare-interpretability trade-off between decomposed and single-model approaches.

## Key Findings

| Finding | Detail |
|---------|--------|
| Value-weighted targeting outperforms CATE-only under misspecification (in simulation) | 6,955 vs 6,700 welfare (*p* < 0.0001); magnitude is DGP-specific |
| Calibration recovers *g* = 1 across all scenarios | Consistent with the value-conservatism trade-off (Proposition 3) |
| Value-weighting advantage is budget-dependent | Strongest at moderate-to-generous budgets; at tight budgets, pure CATE targeting may be preferable |
| Welfare-interpretability trade-off is quantifiable (in simulation) | S-Learner achieves 46% higher welfare than decomposed strategies; magnitude is DGP-specific |

## Repository Structure

```
docs/
  manuscript_v38.tex        LaTeX source (current version)
  manuscript_v38.pdf        Compiled manuscript
src/
  simulation_v38.py         Simulation code (all scenarios, calibration, robustness checks)
previous/                   Archived drafts, reviews, and data (not tracked)
```

## Building the PDF

Requires [tectonic](https://tectonic-typesetting.github.io/) or a standard LaTeX distribution (pdflatex).

```bash
cd docs && tectonic manuscript_v38.tex
```

## Citation

```bibtex
@unpublished{stent2026bonus,
  title   = {Who Gets the Bonus? Welfare-Optimal Treatment Targeting with Value Weights and Risk Modulation},
  author  = {Stent, Mark},
  year    = {2026},
  note    = {Working paper}
}
```

## License

All rights reserved. This repository is provided for review and reference purposes.
