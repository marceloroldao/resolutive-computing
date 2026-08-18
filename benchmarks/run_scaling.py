"""Dimensional scaling experiment for reproducible optimizer comparisons."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from resolutive.benchmarks.functions import DEFAULT_BENCHMARKS
from resolutive.optimization.baselines import DifferentialEvolution, RandomSearch, SimulatedAnnealing
from resolutive.optimization.v2 import ResolutiveV2
from resolutive.optimization.v5 import ResolutiveV5


def run(dimensions: list[int], budget_per_dimension: int, seeds: int, output: Path) -> None:
    optimizers = {
        "RandomSearch": RandomSearch(),
        "SimulatedAnnealing": SimulatedAnnealing(),
        "DifferentialEvolution": DifferentialEvolution(),
        "RO-V2": ResolutiveV2(),
        "RO-V5": ResolutiveV5(),
    }
    rows = []
    for dimension in dimensions:
        budget = budget_per_dimension * dimension
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
                        "budget_per_dimension": budget_per_dimension,
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
    parser.add_argument("--dimensions", type=int, nargs="+", default=[10, 20, 30, 50])
    parser.add_argument("--budget-per-dimension", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/scaling.csv"))
    args = parser.parse_args()
    run(args.dimensions, args.budget_per_dimension, args.seeds, args.output)
