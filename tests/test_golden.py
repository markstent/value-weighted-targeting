"""Unrounded per-replication regression lock (3 reps, n=2000, kappa=0.30).

Regenerate with tests/golden/generate.py only for an intentional numerical change.
"""
import json
from pathlib import Path

import pytest

from core.dgp import generate_null
from study_main import SEED_BASE, run_main, run_single_simulation

GOLDEN = json.loads((Path(__file__).parent / 'golden' / 'main_smoke.json').read_text())
DGP_FUNCS = {'main': None, 'null': generate_null}


def _per_rep(dgp_func):
    got = {}
    for run in range(3):
        results = run_single_simulation(
            SEED_BASE + run * 7, budget_frac=0.30, dgp_func=dgp_func,
            n_players=2000)[0]
        for name, r in results.items():
            d = got.setdefault(name, {'welfare': [], 'risk_exposure': []})
            d['welfare'].append(r['welfare'])
            d['risk_exposure'].append(r['risk_exposure'])
    return got


@pytest.mark.parametrize('case', sorted(DGP_FUNCS))
def test_smoke_matches_golden(case):
    got = _per_rep(DGP_FUNCS[case])
    expected = GOLDEN[case]
    assert set(got) == set(expected)
    for strategy, exp in expected.items():
        for key in ('welfare', 'risk_exposure'):
            assert got[strategy][key] == pytest.approx(exp[key], abs=1e-9), (
                case, strategy, key)


def test_run_main_aggregates_golden_main_welfare():
    agg = run_main(reps=3, n=2000, kappa=0.30)
    for strategy, exp in GOLDEN['main'].items():
        mean = sum(exp['welfare']) / 3
        assert agg[strategy]['welfare_mean'] == pytest.approx(mean, abs=0.5 + 1e-9)
