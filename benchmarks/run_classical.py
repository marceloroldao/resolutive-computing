"""Reproducible classical benchmark runner for Resolutive Optimizers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from resolutive.benchmarks.functions import DEFAULT_BENCHMARKS
from resolutive.optimization.v2 import ResolutiveV2
from resolutive.optimization.v5 import ResolutiveV5


def run(dimension: int, budget: int, seeds: int, output: Path) -> None:
    optimizers = {"RO-V2": ResolutiveV2(), "RO-V5": ResolutiveV5()}
    rows = []
    for benchmark_name, (objective, bounds) in DEFAULT_BENCHMARKS.items():
        for optimizer_name, optimizer in optimizers.items():
            values = []
            evaluations = []
            for seed in range(seeds):
                result = optimizer.minimize(
                    objective,
                    dimension=dimension,
                    bounds=bounds,
                    budget=budget,
                    seed=seed,
                )
                values.append(result.fun)
                evaluations.append(result.evaluations)
            rows.append(
                {
                    "benchmark": benchmark_name,
                    "optimizer": optimizer_name,
                    "dimension": dimension,
                    "budget": budget,
                    "seeds": seeds,
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if seeds > 1 else 0.0,
                    "best": float(np.min(values)),
                    "worst": float(np.max(values)),
                    "max_evaluations_used": int(np.max(evaluations)),
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--budget", type=int, default=6000)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("results/classical.csv"))
    args = parser.parse_args()
    run(args.dimension, args.budget, args.seeds, args.output)
