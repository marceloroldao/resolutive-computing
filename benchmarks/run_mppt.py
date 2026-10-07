"""Reproducible synthetic MPPT comparison across optimizers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from resolutive.applications.energy import mppt_clear_sky_cost, mppt_partial_shading_cost
from resolutive.optimization.baselines import DifferentialEvolution, RandomSearch, SimulatedAnnealing
from resolutive.optimization.v2 import ResolutiveV2
from resolutive.optimization.v5 import ResolutiveV5

PROBLEMS = {
    "clear_sky": mppt_clear_sky_cost,
    "partial_shading": mppt_partial_shading_cost,
}

OPTIMIZERS = {
    "RandomSearch": RandomSearch(),
    "SimulatedAnnealing": SimulatedAnnealing(),
    "DifferentialEvolution": DifferentialEvolution(),
    "RO-V2": ResolutiveV2(population=16),
    "RO-V5": ResolutiveV5(population=20),
}


def run(budget: int, seeds: int, output: Path) -> None:
    rows = []
    for problem_name, objective in PROBLEMS.items():
        for optimizer_name, optimizer in OPTIMIZERS.items():
            values = []
            evaluations = []
            for seed in range(seeds):
                result = optimizer.minimize(
                    objective,
                    dimension=1,
                    bounds=(-1.0, 1.0),
                    budget=budget,
                    seed=seed,
                )
                values.append(result.fun)
                evaluations.append(result.evaluations)
            rows.append({
                "problem": problem_name,
                "optimizer": optimizer_name,
                "budget": budget,
                "seeds": seeds,
                "median_cost": float(np.median(values)),
                "median_power": float(-np.median(values)),
                "mean_cost": float(np.mean(values)),
                "std_cost": float(np.std(values, ddof=1)) if seeds > 1 else 0.0,
                "best_power": float(-np.min(values)),
                "worst_power": float(-np.max(values)),
                "max_evaluations_used": int(np.max(evaluations)),
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("results/mppt.csv"))
    args = parser.parse_args()
    run(args.budget, args.seeds, args.output)
