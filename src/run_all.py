"""One-command pipeline: every study, then the figures, then a run manifest.

Usage:  python src/run_all.py [--smoke] [--jobs N] [--only study,...] [--reps R] [--n-players N]

Studies (study_main, study_harm, study_budget, study_learners, study_robustness) run as
subprocesses with the same run-size flags; up to N run in parallel.  Each logs to
outputs/logs/<study>.log.  make_figures.py runs afterwards if every study succeeded.
Exits non-zero, naming the failed step, if anything fails.  outputs/run_manifest.json
records the git SHA and dirty flag, python and package versions, run-size args, per-step
wall time and exit code, the JSON outputs with their sha256, the figures, and the sha256
of config/tuned_params.json.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'src'
OUTPUT_DIR = REPO / 'outputs'
LOG_DIR = OUTPUT_DIR / 'logs'
FIGURE_DIR = REPO / 'docs' / 'figures'
SMOKE_FIGURE_DIR = OUTPUT_DIR / 'figures_smoke'   # smoke runs never touch the manuscript figures
TUNED_CONFIG = REPO / 'config' / 'tuned_params.json'

STUDIES = ('study_main', 'study_harm', 'study_budget', 'study_learners', 'study_robustness')
OUTPUT_BASENAMES = {
    'study_main': 'main', 'study_harm': 'harm_sweep', 'study_budget': 'budget_2x2',
    'study_learners': 'learners', 'study_robustness': 'robustness',
}
PACKAGES = ('numpy', 'scipy', 'scikit-learn', 'matplotlib', 'joblib', 'pytest')


def output_name(study, smoke):
    """JSON file a study writes: <base>.json, or <base>_smoke.json under --smoke."""
    return OUTPUT_BASENAMES[study] + ('_smoke' if smoke else '') + '.json'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def size_flags(args):
    flags = []
    if args.smoke:
        flags.append('--smoke')
    if args.reps is not None:
        flags += ['--reps', str(args.reps)]
    if args.n_players is not None:
        flags += ['--n-players', str(args.n_players)]
    return flags


def _git(*cmd):
    try:
        return subprocess.run(['git', *cmd], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_state():
    sha = _git('rev-parse', 'HEAD')
    status = _git('status', '--porcelain', '--', '.', ':(exclude)outputs')
    return {'sha': sha, 'dirty': None if status is None else bool(status)}


def package_versions():
    out = {}
    for pkg in PACKAGES:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def run_step(name, script, flags):
    """Run src/<script> with flags, logging to outputs/logs/<name>.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f'{name}.log'
    t0 = time.perf_counter()
    with open(log, 'w') as fp:
        proc = subprocess.run([sys.executable, str(SRC / script), *flags], cwd=REPO,
                              stdout=fp, stderr=subprocess.STDOUT)
    return {'name': name, 'exit_code': proc.returncode,
            'wall_seconds': round(time.perf_counter() - t0, 2),
            'log': str(log.relative_to(REPO))}


def run_studies(studies, flags, jobs):
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [pool.submit(run_step, s, f'{s}.py', flags) for s in studies]
        return [f.result() for f in futures]


def build_manifest(args, steps, studies, git, started):
    json_files = {}
    for s in studies:
        path = OUTPUT_DIR / output_name(s, args.smoke)
        if path.exists():
            json_files[path.name] = sha256(path)
    fig_dir = SMOKE_FIGURE_DIR if args.smoke else FIGURE_DIR
    figures = {str(p.relative_to(REPO)): sha256(p) for p in sorted(fig_dir.glob('*.pdf'))} \
        if any(st['name'] == 'make_figures' and st['exit_code'] == 0 for st in steps) else {}
    return {
        'git': git,
        'python': platform.python_version(),
        'packages': package_versions(),
        'args': {'smoke': args.smoke, 'jobs': args.jobs, 'only': list(studies),
                 'reps': args.reps, 'n_players': args.n_players},
        'started_utc': started,
        'steps': steps,
        'json_files': json_files,
        'figures': figures,
        'tuned_params_sha256': sha256(TUNED_CONFIG) if TUNED_CONFIG.exists() else None,
    }


def parse_only(text):
    names = [t.strip() for t in text.split(',') if t.strip()]
    names = [n if n.startswith('study_') else f'study_{n}' for n in names]
    unknown = [n for n in names if n not in STUDIES]
    if unknown:
        raise argparse.ArgumentTypeError(f'unknown study {unknown}; choose from {list(STUDIES)}')
    return [s for s in STUDIES if s in names]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--smoke', action='store_true', help='quick run (3 reps, n=2000)')
    parser.add_argument('--reps', type=int, default=None)
    parser.add_argument('--n-players', type=int, default=None)
    parser.add_argument('--jobs', type=int, default=1, help='studies to run in parallel')
    parser.add_argument('--only', type=parse_only, default=list(STUDIES),
                        help='comma-separated subset of studies (e.g. main,harm)')
    args = parser.parse_args(argv)

    git = git_state()
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    flags = size_flags(args)
    t0 = time.perf_counter()
    steps = run_studies(args.only, flags, args.jobs)
    failed = [s['name'] for s in steps if s['exit_code'] != 0]
    if not failed:
        steps.append(run_step('make_figures', 'make_figures.py', ['--smoke'] if args.smoke else []))
        failed = [s['name'] for s in steps if s['exit_code'] != 0]

    manifest = build_manifest(args, steps, args.only, git, started)
    manifest['total_wall_seconds'] = round(time.perf_counter() - t0, 2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / 'run_manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')

    for s in steps:
        print(f"{s['name']:<18} exit={s['exit_code']} {s['wall_seconds']:>8.1f}s  {s['log']}")
    print(f'manifest: {path}')
    if failed:
        print(f"FAILED: {', '.join(failed)} (see outputs/logs/)", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
