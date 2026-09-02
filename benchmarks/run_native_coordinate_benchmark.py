"""Compare deterministic Python and C++ coordinate-polish kernels.

This benchmark is intentionally narrow: it measures implementation overhead for
one hot deterministic kernel while holding objective, starting point, budget,
step schedule, and acceptance rule constant. It is not a full optimizer claim.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np


def ackley(x: np.ndarray) -> float:
    n = float(x.size)
    return float(-20.0 * np.exp(-0.2 * np.sqrt(np.sum(x * x) / n)) - np.exp(np.sum(np.cos(2.0 * np.pi * x)) / n) + 20.0 + math.e)


def initial_point(dimension: int, lo: float, hi: float, seed: int) -> np.ndarray:
    span = hi - lo
    return np.asarray([lo + span * (((i * 37 + seed * 101 + 17) % 1000) / 999.0) for i in range(dimension)], dtype=float)


def python_polish(dimension: int, budget: int, seed: int) -> tuple[float, int]:
    lo, hi = -32.768, 32.768
    span = hi - lo
    best_x = initial_point(dimension, lo, hi, seed)
    best_f = ackley(best_x)
    used = 1
    step = 0.000625 * span
    while used + 2 * dimension <= budget and step > 1e-13 * span:
        improved = False
        for axis in range(dimension):
            for sign in (-1.0, 1.0):
                cand = best_x.copy()
                cand[axis] = np.clip(cand[axis] + sign * step, lo, hi)
                val = ackley(cand)
                used += 1
                if val < best_f:
                    best_x, best_f = cand, val
                    improved = True
        if not improved:
            step *= 0.25
    return float(best_f), int(used)


def compile_cpp(root: Path, output: Path) -> None:
    source = root / "native" / "coordinate_polish_bench.cpp"
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [os.environ.get("CXX", "g++"), "-O3", "-DNDEBUG", "-std=c++20", str(source), "-o", str(output)]
    subprocess.run(cmd, check=True)


def run_cpp(binary: Path, dimension: int, budget: int, seed: int, repeats: int) -> tuple[float, int, float]:
    cp = subprocess.run([str(binary), str(dimension), str(budget), str(seed), str(repeats)], check=True, text=True, capture_output=True)
    m = re.fullmatch(r"fun=([^ ]+) evaluations=(\d+) seconds=([^ ]+) repeats=(\d+)\n?", cp.stdout)
    if not m:
        raise RuntimeError(f"unexpected C++ output: {cp.stdout!r}")
    return float(m.group(1)), int(m.group(2)), float(m.group(3))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dimensions", type=int, nargs="+", default=[10, 30, 100])
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("results/native-coordinate/summary.csv"))
    a = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    binary = root / "build" / "coordinate_polish_bench"
    compile_cpp(root, binary)

    rows: list[dict[str, object]] = []
    for dim in a.dimensions:
        budget = max(1000, 300 * dim)
        t0 = time.perf_counter()
        py_fun = py_evals = None
        for _ in range(a.repeats):
            py_fun, py_evals = python_polish(dim, budget, a.seed)
        py_seconds = time.perf_counter() - t0

        cpp_fun, cpp_evals, cpp_seconds = run_cpp(binary, dim, budget, a.seed, a.repeats)
        assert py_fun is not None and py_evals is not None
        fun_delta = abs(py_fun - cpp_fun)
        parity = py_evals == cpp_evals and fun_delta <= 1e-12 * max(1.0, abs(py_fun), abs(cpp_fun))
        speedup = py_seconds / cpp_seconds if cpp_seconds > 0 else float("inf")
        row = {
            "dimension": dim,
            "budget": budget,
            "repeats": a.repeats,
            "python_fun": py_fun,
            "cpp_fun": cpp_fun,
            "fun_abs_delta": fun_delta,
            "python_evaluations": py_evals,
            "cpp_evaluations": cpp_evals,
            "parity": parity,
            "python_seconds": py_seconds,
            "cpp_seconds": cpp_seconds,
            "speedup_python_over_cpp": speedup,
        }
        rows.append(row)
        print(f"{dim}D parity={parity} evals={py_evals} python={py_seconds:.6f}s cpp={cpp_seconds:.6f}s speedup={speedup:.2f}x delta={fun_delta:.3e}")
        if not parity:
            raise SystemExit(f"parity failure at {dim}D")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
