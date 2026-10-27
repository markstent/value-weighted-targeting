"""One-command pipeline: every study, then the figures, then a run manifest.

Usage:  python src/run_all.py [--smoke] [--jobs N] [--only study,...] [--figures-only] [--reps R] [--n-players N]

Studies (study_main, study_harm, study_budget, study_learners, study_robustness) run as
subprocesses with the same run-size flags; up to N run in parallel.  Each logs to
outputs/logs/<study>.log.  make_figures.py runs afterwards if every study succeeded.
Exits non-zero, naming the failed step, if anything fails.  outputs/run_manifest.json
records the git SHA and dirty flag, python and package versions, run-size args, per-step
wall time and exit code, the JSON outputs with their sha256, the figures, and the sha256
of the tuned-parameter file the run actually read (config/tuned_params.json at full size, outputs/tuned_params_n<N>.json otherwise).
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


MISSING_EXIT = 97   # exit code recorded when a step exits 0 but an expected output is missing


def check_outputs(expected):
    """Return an error string naming expected files that do not exist, or None.

    Existence alone proves freshness: run_step deletes every expected output before the
    step starts, so a file that exists afterwards was written by this step.  No mtime
    comparison is made, because coarse timestamp resolution (HFS+, FAT, NFS/SMB) or clock
    skew would make it fail falsely, most likely for fast steps."""
    missing = [path.name for path in expected if not path.exists()]
    return '; '.join(f'{n} missing' for n in missing) or None


def run_step(name, script, flags, expected=()):
    """Run src/<script> with flags, logging to outputs/logs/<name>.log.

    Every expected output is deleted first.  After a zero exit each must exist, otherwise
    the step is recorded as failed (exit 97) with an 'error' naming the file, so a study
    that writes nothing is never certified."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f'{name}.log'
    for path in expected:
        path.unlink(missing_ok=True)
    t0 = time.perf_counter()
    with open(log, 'w') as fp:
        proc = subprocess.run([sys.executable, str(SRC / script), *flags], cwd=REPO,
                              stdout=fp, stderr=subprocess.STDOUT)
    code, error = proc.returncode, None
    if code == 0:
        error = check_outputs(expected)
        if error:
            code = MISSING_EXIT
    rec = {'name': name, 'exit_code': code,
           'wall_seconds': round(time.perf_counter() - t0, 2),
           'log': str(log.relative_to(REPO))}
    if error:
        rec['error'] = f'expected output {error}'
    return rec


def expected_outputs(study, smoke):
    return [OUTPUT_DIR / output_name(study, smoke)]


def figure_dir(smoke):
    return SMOKE_FIGURE_DIR if smoke else FIGURE_DIR


def expected_figures(smoke):
    from make_figures import FIGURE_NAMES
    return [figure_dir(smoke) / n for n in FIGURE_NAMES]


def effective_n_players(args):
    """Population size the studies will use (decides which tuned-params file is read)."""
    from core.io import SMOKE_N_PLAYERS
    from study_main import N_PLAYERS
    if args.n_players is not None:
        return args.n_players
    return SMOKE_N_PLAYERS if args.smoke else N_PLAYERS


def tuned_params_file(args):
    from study_learners import tuned_path
    return tuned_path(effective_n_players(args))


# study_robustness reads (and, at a new run size, creates) the tuned-parameter file that
# study_learners writes, so when both run it starts after study_learners finishes: the
# tuning pilot is run once instead of twice in parallel.
# Studies with their own worker pool (tuning pilot, scenarios) get --jobs too.
INNER_PARALLEL = ('study_learners', 'study_robustness')
AFTER = {'study_robustness': 'study_learners'}


def run_studies(studies, flags, jobs, smoke=False):
    """Run studies with up to `jobs` at once, honouring AFTER; results in input order."""
    import threading
    done = {s: threading.Event() for s in studies}

    def work(s):
        dep = AFTER.get(s)
        if dep in done:
            done[dep].wait()
        try:
            extra = ['--jobs', str(jobs)] if s in INNER_PARALLEL and jobs > 1 else []
            return run_step(s, f'{s}.py', flags + extra, expected_outputs(s, smoke))
        finally:
            done[s].set()

    # Submit dependencies first so a waiting study never holds the last worker slot.
    order = sorted(studies, key=lambda s: s in AFTER)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {s: pool.submit(work, s) for s in order}
        return [futures[s].result() for s in studies]


def build_manifest(args, steps, studies, git, started):
    """Hash only the outputs of steps that passed their output check."""
    ok = {st['name'] for st in steps if st['exit_code'] == 0}
    json_files = {}
    for study in studies:
        path = OUTPUT_DIR / output_name(study, args.smoke)
        if (study in ok or getattr(args, 'figures_only', False)) and path.exists():
            json_files[path.name] = sha256(path)
    figures = {}
    if 'make_figures' in ok:
        figures = {str(p.relative_to(REPO)): sha256(p) for p in expected_figures(args.smoke)}
    tuned = tuned_params_file(args)
    return {
        'git': git,
        'python': platform.python_version(),
        'packages': package_versions(),
        'args': {'smoke': args.smoke, 'jobs': args.jobs, 'only': list(studies),
                 'reps': args.reps, 'n_players': args.n_players,
                 'figures_only': getattr(args, 'figures_only', False)},
        'started_utc': started,
        'steps': steps,
        'json_files': json_files,
        'figures': figures,
        'tuned_params': {
            'path': str(tuned.relative_to(REPO)) if tuned.is_relative_to(REPO) else str(tuned),
            'sha256': sha256(tuned) if tuned.exists() else None},
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
    parser.add_argument('--figures-only', action='store_true',
                        help='rebuild figures from existing outputs without running studies')
    args = parser.parse_args(argv)

    git = git_state()
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    flags = size_flags(args)
    t0 = time.perf_counter()
    suffix = '_smoke' if args.smoke else ''
    fig_inputs = [OUTPUT_DIR / f'main{suffix}.json', OUTPUT_DIR / f'harm_sweep{suffix}.json']
    steps = []
    if args.figures_only:
        for path in fig_inputs:
            if path.exists():
                print(f'figure input {path.name}: modified '
                      f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))}')
            else:
                print(f'figure input {path.name}: MISSING')
        args.only = ['study_main', 'study_harm']
    else:
        steps = run_studies(args.only, flags, args.jobs, args.smoke)
    failed = [s['name'] for s in steps if s['exit_code'] != 0]
    if failed:
        pass
    elif args.figures_only or {'study_main', 'study_harm'} <= set(args.only):
        steps.append(run_step('make_figures', 'make_figures.py',
                              ['--smoke'] if args.smoke else [], expected_figures(args.smoke)))
        failed = [s['name'] for s in steps if s['exit_code'] != 0]
    else:
        print('make_figures skipped: it needs study_main and study_harm to have run in '
              'this invocation (use --figures-only to rebuild from existing outputs)')

    manifest = build_manifest(args, steps, args.only, git, started)
    manifest['total_wall_seconds'] = round(time.perf_counter() - t0, 2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / 'run_manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')

    for s in steps:
        print(f"{s['name']:<18} exit={s['exit_code']} {s['wall_seconds']:>8.1f}s  {s['log']}"
              + (f"  ERROR: {s['error']}" if s.get('error') else ''))
    print(f'manifest: {path}')
    if failed:
        detail = '; '.join(f"{s['name']}: {s['error']}" for s in steps if s.get('error'))
        print(f"FAILED: {', '.join(failed)} (see outputs/logs/)" + (f' [{detail}]' if detail else ''),
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
