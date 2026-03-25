<div align="center">

# A Causal Framework for Churn-Regularised Bonus Targeting in Online Gaming

Mark Stent | March 2026

---

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

## Overview

A modular, budget-constrained framework for targeting player retention bonuses using causal uplift modelling, lifetime value weighting, and a novel churn regularisation function.

The composite scoring rule $s_i = \hat{\tau}_i \times \text{LTV}_i \times f(p_i^{\text{churn}})$ combines three independently interpretable components into a single targeting priority. The churn regularisation function $f$ implements an explicit, operator-configurable trade-off between expected value and targeting conservatism.

## Key Results

**Simulation Study** (10,000 players, 30% budget, 50 replications):

| Strategy | ATE (pp) | Incr. Value | Persuadable % |
|---|---|---|---|
| Random | 5.5 | $29,299 | 15.1% |
| Response Model | 3.1 | $24,360 | 15.6% |
| **T-Learner pathway** | | | |
| Uplift Only | 10.6 | $67,097 | 27.8% |
| Uplift x LTV | 9.7 | $74,181 | 27.6% |
| Full Framework (T) | 10.2 | $62,361 | 22.5% |
| **S-Learner pathway** | | | |
| S-Learner | 12.5 | $77,337 | 31.8% |
| S-Learner x LTV | 11.4 | $88,566 | 31.5% |
| Full Framework (S) | 10.7 | $66,584 | 24.2% |
| **X-Learner pathway** | | | |
| X-Learner | 11.8 | $73,208 | 30.1% |
| X-Learner x LTV | 10.4 | $84,051 | 29.5% |
| Full Framework (X) | 10.7 | $65,482 | 23.8% |

The framework is meta-learner agnostic: T-Learner, S-Learner, and X-Learner CATE estimates all work within the composite rule. The ablation chains show that each component adds distinct value: LTV weighting increases incremental value while churn regularisation trades value for targeting conservatism (16-25% reduction).

## Repository Structure

```
├── docs/
│   ├── manuscript_v12.2.tex         # Current manuscript (LaTeX)
│   ├── manuscript_v12.2.pdf         # Compiled PDF
│   └── archive/                     # Previous versions and reviews
├── src/
│   ├── simulation_analysis.py       # Main simulation (8 strategies, 50 runs)
│   ├── supplementary_analysis.py    # X-Learner, churn sensitivity, robustness
│   └── hillstrom_analysis.py        # Cross-domain validation
├── tests/                           # Unit and regression tests
├── notebooks/                       # Jupyter notebooks
├── figures/                         # Generated figures
├── outputs/                         # Simulation results (JSON)
├── Makefile
└── README.md
```

## Build

```bash
make simulation      # Run main simulation analysis
make supplementary   # Run supplementary analyses (X-Learner, sensitivity)
make hillstrom       # Run Hillstrom cross-domain validation
make test            # Run test suite
make manuscript      # Compile LaTeX to PDF
make all             # Run everything
```

## Data Sources

| Dataset | Records | Source |
|---|---|---|
| Simulation | 10,000 | Synthetic DGP with known potential outcomes |
| Hillstrom | 64,000 | [MineThatData](http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv) |

## License

MIT
