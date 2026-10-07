"""Dimensional scaling experiment for reproducible optimizer comparisons."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cma
import numpy as np

from resolutive.benchmarks.functions import DEFAULT_BENCHMARKS
from resolutive.optimization.baselines import DifferentialEvolution, RandomSearch, SimulatedAnnealing
from resolutive.optimization.hybrid_multires import ResolutiveHybridMultiResolution
from resolutive.optimization.hybrid_regime import ResolutiveHybridRegime
from resolutive.optimization.v2 import ResolutiveV2
from resolutive.optimization.v5 import ResolutiveV5


def _cma_es(objective, dimension, bounds, budget, seed):
    lo, hi = bounds
    x0 = np.full(dimension, (lo + hi) / 2.0)
    sigma0 = (hi - lo) / 4.0
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        "bounds": [lo, hi], "seed": seed + 1, "verbose": -9,
        "verb_disp": 0, "maxfevals": budget,
    })
    used = 0
    best_x = x0.copy()
    best = float("inf")
    while not es.stop() and used < budget:
        xs = es.ask()
        xs = xs[: max(0, budget - used)]
        if not xs:
            break
        ys = [float(objective(np.asarray(x))) for x in xs]
        used += len(ys)
        j = int(np.argmin(ys))
        if ys[j] < best:
            best = float(ys[j])
            best_x = np.asarray(xs[j], dtype=float).copy()
        es.tell(xs, ys)
    return best_x, best, used


def run(dimensions: list[int], budget_per_dimension: int, seeds: int, output: Path) -> None:
    optimizers = {
        "CMA-ES(pycma)": None,
        "RandomSearch": RandomSearch(),
        "SimulatedAnnealing": SimulatedAnnealing(),
        "DifferentialEvolution": DifferentialEvolution(),
        "RO-V2": ResolutiveV2(),
        "RO-V5": ResolutiveV5(),
        "RO-Hybrid-exp": ResolutiveHybridRegime(),
        "RO-Hybrid-Multires-exp": ResolutiveHybridMultiResolution(),
    }
    rows = []
    for dimension in dimensions:
        budget = budget_per_dimension * dimension
        for benchmark_name, (objective, bounds) in DEFAULT_BENCHMARKS.items():
            for optimizer_name, optimizer in optimizers.items():
                values = []
                evaluations = []
                for seed in range(seeds):
                    if optimizer_name.startswith("CMA-ES"):
                        _x, value, used = _cma_es(objective, dimension, bounds, budget, seed)
                    else:
                        result = optimizer.minimize(
                            objective,
                            dimension=dimension,
                            bounds=bounds,
                            budget=budget,
                            seed=seed,
                        )
                        value, used = result.fun, result.evaluations
                    values.append(float(value))
                    evaluations.append(int(used))
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
