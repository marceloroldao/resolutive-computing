"""Computational-efficiency benchmark for Resolutive Computing.

Measures quality and resource cost separately from the release quality benchmark.
Metrics are collected per optimizer/run under the same benchmark, dimension,
seed and strict objective-call budget:

- final best objective value
- objective evaluations used
- first evaluation reaching absolute targets 1e-2 and 1e-6
- wall-clock time (perf_counter)
- CPU process time
- sampled process RSS baseline/peak/delta

Wall time and RSS are infrastructure-sensitive. They are intended for
same-runner comparative evidence, not hardware-independent claims.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psutil

from resolutive.benchmarks.functions import DEFAULT_BENCHMARKS
from run_release_benchmark import METHODS, default_budget

TARGETS = (1e-2, 1e-6)


@dataclass
class TrackedObjective:
    objective: object
    evaluations: int = 0
    hits: dict[float, int | None] = field(default_factory=lambda: {target: None for target in TARGETS})

    def __call__(self, x) -> float:
        value = float(self.objective(np.asarray(x, dtype=float)))
        self.evaluations += 1
        for target in TARGETS:
            if self.hits[target] is None and value <= target:
                self.hits[target] = self.evaluations
        return value


class RSSSampler:
    def __init__(self, interval_s: float = 0.002):
        self.process = psutil.Process(os.getpid())
        self.interval_s = interval_s
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        try:
            self.peak = max(self.peak, self.process.memory_info().rss)
        except psutil.Error:
            pass


def run(*, dimensions: list[int], seeds: int, raw_output: Path, summary_output: Path, budget_override: int | None = None) -> None:
    rows: list[dict[str, object]] = []

    for dimension in dimensions:
        budget = budget_override if budget_override is not None else default_budget(dimension)
        for benchmark_name, (objective, bounds) in DEFAULT_BENCHMARKS.items():
            for seed in range(seeds):
                for optimizer_name, runner in METHODS.items():
                    gc.collect()
                    tracked = TrackedObjective(objective)
                    sampler = RSSSampler()
                    cpu_start = time.process_time()
                    wall_start = time.perf_counter()
                    with sampler:
                        value, used = runner(tracked, dimension, bounds, budget, seed)
                    wall_s = time.perf_counter() - wall_start
                    cpu_s = time.process_time() - cpu_start
                    rows.append({
                        "benchmark": benchmark_name,
                        "dimension": dimension,
                        "budget": budget,
                        "seed": seed,
                        "optimizer": optimizer_name,
                        "fun": value,
                        "evaluations": used,
                        "tracked_evaluations": tracked.evaluations,
                        "evals_to_1e-2": tracked.hits[1e-2] if tracked.hits[1e-2] is not None else "",
                        "evals_to_1e-6": tracked.hits[1e-6] if tracked.hits[1e-6] is not None else "",
                        "wall_time_s": wall_s,
                        "cpu_time_s": cpu_s,
                        "rss_baseline_bytes": sampler.baseline,
                        "rss_peak_bytes": sampler.peak,
                        "rss_delta_bytes": max(0, sampler.peak - sampler.baseline),
                    })

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    with raw_output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, int, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["benchmark"]), int(row["dimension"]), int(row["budget"]), str(row["optimizer"]))
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for (benchmark, dimension, budget, optimizer), group in grouped.items():
        def values(name: str) -> np.ndarray:
            return np.asarray([float(row[name]) for row in group], dtype=float)

        wall = values("wall_time_s")
        cpu = values("cpu_time_s")
        rss = values("rss_delta_bytes")
        evals = values("evaluations")
        fun = values("fun")

        def target_stats(name: str) -> tuple[int, float | str]:
            reached = [float(row[name]) for row in group if row[name] != ""]
            return len(reached), (float(np.median(reached)) if reached else "")

        reached_1e2, median_1e2 = target_stats("evals_to_1e-2")
        reached_1e6, median_1e6 = target_stats("evals_to_1e-6")
        summary.append({
            "benchmark": benchmark,
            "dimension": dimension,
            "budget": budget,
            "optimizer": optimizer,
            "seeds": len(group),
            "median_fun": float(np.median(fun)),
            "median_evaluations": float(np.median(evals)),
            "median_wall_time_s": float(np.median(wall)),
            "median_cpu_time_s": float(np.median(cpu)),
            "median_rss_delta_bytes": float(np.median(rss)),
            "reached_1e-2": reached_1e2,
            "median_evals_to_1e-2": median_1e2,
            "reached_1e-6": reached_1e6,
            "median_evals_to_1e-6": median_1e6,
        })

    summary.sort(key=lambda row: (str(row["benchmark"]), int(row["dimension"]), str(row["optimizer"])))
    with summary_output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"platform={platform.platform()} python={platform.python_version()} scenarios={len(grouped)} runs={len(rows)}")
    print(f"raw={raw_output} summary={summary_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 10, 30])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--raw-output", type=Path, default=Path("results/efficiency-benchmark-v1/raw.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/efficiency-benchmark-v1/summary.csv"))
    args = parser.parse_args()
    run(
        dimensions=args.dimensions,
        seeds=args.seeds,
        raw_output=args.raw_output,
        summary_output=args.summary_output,
        budget_override=args.budget,
    )
