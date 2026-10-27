"""Regenerate tests/golden/main_smoke.json (unrounded per-rep values).

Run from repo root: .venv/bin/python tests/golden/generate.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from core.dgp import generate_null  # noqa: E402
from simulation_v38 import SEED_BASE, run_single_simulation  # noqa: E402

REPS, N, KAPPA = 3, 2000, 0.30
CASES = {'main': None, 'null': generate_null}


def collect(dgp_func):
    per = {}
    for run in range(REPS):
        results = run_single_simulation(
            SEED_BASE + run * 7, budget_frac=KAPPA, dgp_func=dgp_func,
            n_players=N)[0]
        for name, r in results.items():
            d = per.setdefault(name, {'welfare': [], 'risk_exposure': []})
            d['welfare'].append(float(r['welfare']))
            d['risk_exposure'].append(float(r['risk_exposure']))
    return per


if __name__ == '__main__':
    out = {case: collect(f) for case, f in CASES.items()}
    (ROOT / 'tests' / 'golden' / 'main_smoke.json').write_text(
        json.dumps(out, indent=2))
