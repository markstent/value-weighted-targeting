"""End-to-end: a fresh clone runs the whole smoke pipeline and the manifest is honest."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import run_all

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_smoke_fresh_clone(tmp_path):
    clone = tmp_path / 'clone'
    subprocess.run(['git', 'clone', '-q', str(ROOT), str(clone)], check=True)
    # reuse this environment's interpreter (the repo .venv when run as .venv/bin/pytest)
    proc = subprocess.run([sys.executable, 'src/run_all.py', '--smoke', '--jobs', '4'],
                          cwd=clone, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]

    manifest = json.loads((clone / 'outputs' / 'run_manifest.json').read_text())
    assert manifest['git']['sha'] and all(s['exit_code'] == 0 for s in manifest['steps'])
    assert {s['name'] for s in manifest['steps']} == set(run_all.STUDIES) | {'make_figures'}
    assert set(manifest['json_files']) == {
        run_all.output_name(s, True) for s in run_all.STUDIES}
    for name, digest in manifest['json_files'].items():
        path = clone / 'outputs' / name
        assert path.exists(), name
        assert run_all.sha256(path) == digest, name
    tuned = manifest['tuned_params']
    assert tuned['path'] == 'outputs/tuned_params_n2000.json'    # smoke runs never read the config
    assert tuned['sha256'] == run_all.sha256(clone / tuned['path'])
    assert len(manifest['figures']) == 4
    for rel, digest in manifest['figures'].items():
        assert (clone / rel).exists(), rel
        assert run_all.sha256(clone / rel) == digest
