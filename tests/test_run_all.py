"""run_all: manifest contents, failure reporting, atomic tuned-param writes."""
import json

import pytest

import run_all


def _args(**kw):
    ns = run_all.argparse.Namespace(smoke=True, reps=None, n_players=None, jobs=1, only=None)
    ns.__dict__.update(kw)
    return ns


def test_output_names():
    assert run_all.output_name('study_main', True) == 'main_smoke.json'
    assert run_all.output_name('study_harm', False) == 'harm_sweep.json'


def test_parse_only_orders_and_validates():
    assert run_all.parse_only('robustness,main') == ['study_main', 'study_robustness']
    with pytest.raises(Exception):
        run_all.parse_only('nope')


def test_size_flags():
    assert run_all.size_flags(_args()) == ['--smoke']
    assert run_all.size_flags(_args(smoke=False, reps=5, n_players=100)) == [
        '--reps', '5', '--n-players', '100']


def test_failure_is_named_and_manifest_written(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_all, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(run_all, 'LOG_DIR', tmp_path / 'logs')

    def fake_step(name, script, flags):
        return {'name': name, 'exit_code': 3 if name == 'study_harm' else 0,
                'wall_seconds': 0.1, 'log': f'outputs/logs/{name}.log'}

    monkeypatch.setattr(run_all, 'run_step', fake_step)
    rc = run_all.main(['--smoke', '--only', 'main,harm', '--jobs', '2'])
    assert rc == 1
    assert 'study_harm' in capsys.readouterr().err
    m = json.loads((tmp_path / 'run_manifest.json').read_text())
    assert {s['name']: s['exit_code'] for s in m['steps']} == {'study_main': 0, 'study_harm': 3}
    assert m['args']['smoke'] is True and set(m['git']) == {'sha', 'dirty'}
    assert 'numpy' in m['packages'] and m['tuned_params_sha256']


def test_manifest_hashes_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, 'OUTPUT_DIR', tmp_path)
    (tmp_path / 'main_smoke.json').write_text('{}')
    m = run_all.build_manifest(_args(), [], ['study_main', 'study_harm'], {}, 't')
    assert m['json_files'] == {'main_smoke.json': run_all.sha256(tmp_path / 'main_smoke.json')}


def test_tuned_params_write_is_atomic(tmp_path, monkeypatch):
    import study_learners
    monkeypatch.setattr(study_learners, 'tuned_path', lambda n: tmp_path / 't.json')
    monkeypatch.setattr(study_learners, 'tune_all',
                        lambda n, j: {'_meta': study_learners.make_meta(n),
                                      'learners': {l: {'params': {}} for l in study_learners.COMPARED_LEARNERS}})
    study_learners.load_or_tune(n_players=study_learners.N_PLAYERS)
    assert (tmp_path / 't.json').exists()
    assert not list(tmp_path.glob('*.tmp'))
