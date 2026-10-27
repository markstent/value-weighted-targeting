"""study_main: oracle tables folded in, ddof=1 descriptive SDs, --smoke wiring, no legacy names."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import study_main
from core.io import load_json

ROOT = Path(__file__).resolve().parents[1]


def test_sd_is_sample_sd():
    x = [1.0, 2.0, 4.0, 8.0]
    assert study_main._sd(x) == pytest.approx(np.std(x, ddof=1))
    assert study_main._sd(x, 1) == round(np.std(x, ddof=1), 1)
    assert study_main._sd([3.0]) is None


def test_aggregate_uses_ddof1():
    runs = [{'ate': a, 'welfare': w, 'risk_exposure': 0.5, 'mean_churn_mod': 0.1,
             'mean_ltv_cr': 1.0} for a, w in [(1, 1000), (2, 1500), (4, 4000)]]
    agg = study_main._aggregate_results({'S': runs}, ['S'])['S']
    assert agg['welfare_sd'] == round(np.std([1000, 1500, 4000], ddof=1))


def test_oracle_tables_match_budget_runs():
    kappa = '0.3'
    res = {nm: [{'welfare': float(w), 'ate': 1.0, 'risk_exposure': 0.4,
                 'mean_churn_mod': 0.2} for w in (10, 20, 40)]
           for nm in study_main.ORACLE_STRATEGIES}
    main_t = study_main._oracle_table(res, 3, 100)
    bud_t = study_main._oracle_budget_table({kappa: res}, 3, 100)
    for nm in study_main.ORACLE_STRATEGIES:
        assert main_t['strategies'][nm]['welfare_sd'] == pytest.approx(np.std([10, 20, 40], ddof=1))
        assert bud_t['strategies'][nm][kappa]['welfare_mean'] == pytest.approx(70 / 3)


def test_smoke_cli_writes_main_json_with_oracles():
    subprocess.run([sys.executable, str(ROOT / 'src' / 'study_main.py'), '--smoke'],
                   check=True, capture_output=True)
    out = load_json('main_smoke.json')
    assert out['design']['n_runs'] == 3 and out['design']['n_players'] == 2000
    assert set(out['oracle_main']['strategies']) == set(study_main.ORACLE_STRATEGIES)
    for kappa in study_main.BUDGET_FRACTIONS:
        for nm in study_main.ORACLE_STRATEGIES:
            assert str(kappa) in out['oracle_budget']['strategies'][nm]
    # oracle_main at kappa=0.30 equals the budget table entry (same runs)
    for nm in study_main.ORACLE_STRATEGIES:
        assert (out['oracle_main']['strategies'][nm]['welfare_mean']
                == pytest.approx(out['oracle_budget']['strategies'][nm]['0.3']['welfare_mean']))
    assert 'tost_cal_vs_cate' in out['cate_misspec_paired']


def test_no_legacy_version_names():
    pat = re.compile(r'v29|v38|v39|v4\d')
    bad = []
    for path in list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'tests').glob('*.py')) + [ROOT / 'README.md']:
        if path == Path(__file__):
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if pat.search(line):
                bad.append(f'{path.name}:{i}')
    assert not bad, bad
    assert not list((ROOT / 'src').glob('*v[0-9]*')) and not (ROOT / 'src' / 'compute_tauv_oracle.py').exists()
