"""--jobs must not change results: n_jobs=1 and n_jobs=2 give exactly equal outputs."""
import json

import numpy as np

import study_learners as sl
import study_main as sm
import study_robustness as sr
from core.io import NumpyEncoder

REPS, N = 2, 800


def test_learners_simulate_same_for_1_and_2_jobs():
    tuned = sl.load_or_tune(n_players=sl.N_PLAYERS)       # tracked frozen config
    a = sl.simulate(REPS, N, tuned, kappas=(0.3,), n_jobs=1)
    b = sl.simulate(REPS, N, tuned, kappas=(0.3,), n_jobs=2)
    assert a['welfare'].keys() == b['welfare'].keys()
    for key in a['welfare']:
        np.testing.assert_array_equal(a['welfare'][key], b['welfare'][key])
    assert a['regret'] == b['regret']
    assert a['diagnostic'] == b['diagnostic']


def test_robustness_scenario_same_for_1_and_2_jobs():
    tuned = {'T': {'max_depth': 2, 'n_estimators': 30, 'learning_rate': 0.1}}
    fn = sr.SCENARIOS['effective_welfare']
    a = fn(REPS, N, tuned, n_jobs=1)
    b = fn(REPS, N, tuned, n_jobs=2)
    assert json.dumps(a, cls=NumpyEncoder, sort_keys=True) == json.dumps(b, cls=NumpyEncoder, sort_keys=True)


def test_main_study_same_for_1_and_2_jobs():
    a = sm.run_study(REPS, N, n_jobs=1)
    b = sm.run_study(REPS, N, n_jobs=2)
    assert json.dumps(a, cls=NumpyEncoder, sort_keys=True) == json.dumps(b, cls=NumpyEncoder, sort_keys=True)
