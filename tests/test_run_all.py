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

    def fake_step(name, script, flags, expected=()):
        return {'name': name, 'exit_code': 3 if name == 'study_harm' else 0,
                'wall_seconds': 0.1, 'log': f'outputs/logs/{name}.log'}

    monkeypatch.setattr(run_all, 'run_step', fake_step)
    rc = run_all.main(['--smoke', '--only', 'main,harm', '--jobs', '2'])
    assert rc == 1
    assert 'study_harm' in capsys.readouterr().err
    m = json.loads((tmp_path / 'run_manifest.json').read_text())
    assert {s['name']: s['exit_code'] for s in m['steps']} == {'study_main': 0, 'study_harm': 3}
    assert m['args']['smoke'] is True and set(m['git']) == {'sha', 'dirty'}
    assert 'numpy' in m['packages']
    assert m['tuned_params']['path'] == 'outputs/tuned_params_n2000.json'


def test_manifest_hashes_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, 'OUTPUT_DIR', tmp_path)
    (tmp_path / 'main_smoke.json').write_text('{}')
    steps = [{'name': 'study_main', 'exit_code': 0}, {'name': 'study_harm', 'exit_code': run_all.MISSING_EXIT}]
    (tmp_path / 'harm_sweep_smoke.json').write_text('{"stale": 1}')   # failed step: not certified
    m = run_all.build_manifest(_args(), steps, ['study_main', 'study_harm'], {}, 't')
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


def test_robustness_waits_for_learners(monkeypatch):
    import time
    events = []

    def fake_step(name, script, flags, expected=()):
        events.append(('start', name))
        time.sleep(0.2 if name == 'study_learners' else 0)
        events.append(('end', name))
        return {'name': name, 'exit_code': 0, 'wall_seconds': 0, 'log': ''}

    monkeypatch.setattr(run_all, 'run_step', fake_step)
    res = run_all.run_studies(['study_learners', 'study_robustness', 'study_main'], [], 3)
    assert [r['name'] for r in res] == ['study_learners', 'study_robustness', 'study_main']
    assert events.index(('end', 'study_learners')) < events.index(('start', 'study_robustness'))


def _fake_src(tmp_path, monkeypatch, body):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'fake.py').write_text(body)
    monkeypatch.setattr(run_all, 'SRC', src)
    monkeypatch.setattr(run_all, 'REPO', tmp_path)
    monkeypatch.setattr(run_all, 'LOG_DIR', tmp_path / 'logs')


def test_study_exiting_zero_without_output_fails(tmp_path, monkeypatch):
    _fake_src(tmp_path, monkeypatch, 'pass\n')
    out = tmp_path / 'main.json'
    rec = run_all.run_step('fake', 'fake.py', [], [out])
    assert rec['exit_code'] == run_all.MISSING_EXIT and 'main.json missing' in rec['error']


def test_preexisting_output_is_deleted_before_the_step(tmp_path, monkeypatch):
    _fake_src(tmp_path, monkeypatch, 'pass\n')
    out = tmp_path / 'main.json'
    out.write_text('{"old": 1}')
    rec = run_all.run_step('fake', 'fake.py', [], [out])
    assert rec['exit_code'] != 0 and not out.exists()


def test_fresh_output_passes(tmp_path, monkeypatch):
    _fake_src(tmp_path, monkeypatch, f'open({str(tmp_path / "main.json")!r}, "w").write("{{}}")\n')
    rec = run_all.run_step('fake', 'fake.py', [], [tmp_path / 'main.json'])
    assert rec['exit_code'] == 0 and 'error' not in rec


def test_run_all_main_fails_on_silent_study(tmp_path, monkeypatch, capsys):
    _fake_src(tmp_path, monkeypatch, 'pass\n')
    monkeypatch.setattr(run_all, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(run_all, 'STUDIES', ('study_main',))
    monkeypatch.setattr(run_all, 'output_name', lambda s, smoke: 'main.json')
    (tmp_path / 'src' / 'study_main.py').write_text('pass\n')
    rc = run_all.main(['--smoke', '--only', 'main'])
    assert rc == 1
    assert 'study_main' in capsys.readouterr().err
    m = json.loads((tmp_path / 'run_manifest.json').read_text())
    assert m['json_files'] == {}


def test_only_without_main_and_harm_skips_figures(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_all, 'OUTPUT_DIR', tmp_path)
    called = []

    def fake_step(name, script, flags, expected=()):
        called.append(name)
        return {'name': name, 'exit_code': 0, 'wall_seconds': 0, 'log': ''}

    monkeypatch.setattr(run_all, 'run_step', fake_step)
    assert run_all.main(['--smoke', '--only', 'main']) == 0
    assert 'make_figures' not in called
    assert 'skipped' in capsys.readouterr().out
    called.clear()
    assert run_all.main(['--smoke', '--only', 'main,harm']) == 0
    assert 'make_figures' in called


def test_figures_only_prints_mtimes_and_runs_only_figures(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_all, 'OUTPUT_DIR', tmp_path)
    (tmp_path / 'main_smoke.json').write_text('{}')
    called = []

    def fake_step(name, script, flags, expected=()):
        called.append(name)
        return {'name': name, 'exit_code': 0, 'wall_seconds': 0, 'log': ''}

    monkeypatch.setattr(run_all, 'run_step', fake_step)
    assert run_all.main(['--smoke', '--figures-only']) == 0
    out = capsys.readouterr().out
    assert called == ['make_figures']
    assert 'main_smoke.json: modified' in out and 'harm_sweep_smoke.json: MISSING' in out
