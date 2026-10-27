import json
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "golden" / "main_smoke.json"


def test_main_smoke_matches_golden():
    from simulation_v38 import run_main

    golden = json.loads(GOLDEN.read_text())
    got = run_main(reps=3, n=2000, kappa=0.30)
    assert set(got) == set(golden)
    for strategy, expected in golden.items():
        for key in ("welfare_mean", "welfare_sd", "risk_mean"):
            assert got[strategy][key] == pytest.approx(expected[key], abs=1e-9), (
                strategy, key)
