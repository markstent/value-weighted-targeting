"""Output location, JSON persistence and shared command-line arguments."""

import json
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

SMOKE_REPS = 3
SMOKE_N_PLAYERS = 2000


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands NumPy scalars and arrays."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(obj, name):
    """Write obj to OUTPUT_DIR/name (creating the directory) and return the path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=2, cls=NumpyEncoder)
    return path


def load_json(name):
    """Read OUTPUT_DIR/name."""
    with open(OUTPUT_DIR / name) as fp:
        return json.load(fp)


def add_common_args(parser):
    """Add --reps, --n-players and --smoke (reps 3, n 2000) to an ArgumentParser."""
    parser.add_argument("--reps", type=int, default=None, help="number of Monte Carlo replications")
    parser.add_argument("--n-players", type=int, default=None, help="population size per replication")
    parser.add_argument(
        "--smoke", action="store_true", help=f"quick run: reps={SMOKE_REPS}, n-players={SMOKE_N_PLAYERS}"
    )


def resolve_run_size(args, default_reps, default_n):
    """Return (reps, n_players) from parsed common args; explicit flags beat --smoke."""
    reps = args.reps
    n = args.n_players
    if args.smoke:
        reps = SMOKE_REPS if reps is None else reps
        n = SMOKE_N_PLAYERS if n is None else n
    return (default_reps if reps is None else reps, default_n if n is None else n)
