import argparse

import numpy as np

from core import io


def test_save_json_creates_dir(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "outputs"
    monkeypatch.setattr(io, "OUTPUT_DIR", target)
    path = io.save_json({"x": np.float64(0.1) + np.float64(0.2), "n": np.int64(3), "a": np.arange(3)}, "out.json")
    assert path.exists() and path.parent == target
    back = io.load_json("out.json")
    assert back["x"] == float(np.float64(0.1) + np.float64(0.2))
    assert back["n"] == 3 and back["a"] == [0, 1, 2]


def test_common_args_smoke():
    parser = argparse.ArgumentParser()
    io.add_common_args(parser)
    assert io.resolve_run_size(parser.parse_args(["--smoke"]), 200, 10000) == (3, 2000)
    assert io.resolve_run_size(parser.parse_args([]), 200, 10000) == (200, 10000)
    assert io.resolve_run_size(parser.parse_args(["--smoke", "--reps", "5"]), 200, 10000) == (5, 2000)
