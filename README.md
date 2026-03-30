# Optimal Risk-Modulated Treatment Targeting Under Budget Constraints

Working paper on budget-constrained treatment targeting with value-weighted uplift and operator-configurable risk modulation.

## Summary

This paper formalises value-weighted uplift targeting (tau x V) as the welfare-maximising rule under a budget constraint, and extends it with a risk-modulation function g that encodes operator preferences at a quantifiable welfare cost. A simulation study across three scenarios (well-specified CATE, misspecified CATE, null DGP) confirms the theoretical predictions and quantifies the welfare-interpretability trade-off between decomposed and single-model approaches.

## Repository Structure

```
docs/                   # Current manuscript versions and PDFs
  manuscript_v37.tex    # Latest manuscript (LaTeX source)
  manuscript_v37.pdf    # Latest compiled PDF
reviews/                # Peer review reports by version
src/                    # Simulation code
outputs/                # Simulation output data (JSON)
previous/               # Archived early versions (v1-v22), legacy code, and data
```

## Current Version

**v37** (2026-03-30) - Latest revision addressing external review feedback:
- Notation consistency fixes (abstract, hat-g clarification)
- Tightened calibration framing (verification, not discovery)
- Budget-dependent tau x V recommendation caveat
- S-Learner hyperparameter qualifier
- Table cross-referencing and caption improvements

## Building

Requires [tectonic](https://tectonic-typesetting.github.io/) or a standard LaTeX distribution:

```bash
cd docs
tectonic manuscript_v37.tex
```

## Key Results

- Value-weighted targeting (tau x V) significantly outperforms CATE-only under misspecification (6,955 vs 6,700; p < 0.0001)
- Calibration recovers g ~ 1 across all scenarios, confirming the value-conservatism trade-off (Proposition 3)
- The tau x V advantage is budget-dependent: strongest at moderate-to-generous budgets (kappa >= 0.20)
- S-Learner achieves 46% higher welfare than decomposed strategies, quantifying the welfare-interpretability trade-off
