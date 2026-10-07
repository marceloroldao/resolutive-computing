"""Compare optimizers on a reproducible PID-tuning application benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from resolutive.applications.control import pid_step_response_cost
from resolutive.optimization.baselines import DifferentialEvolution, RandomSearch, SimulatedAnnealing
from resolutive.optimization.v2 import ResolutiveV2
from resolutive.optimization.v5 import ResolutiveV5

OPTIMIZERS = {
    "RandomSearch": RandomSearch,
    "SimulatedAnnealing": SimulatedAnnealing,
    "DifferentialEvolution": DifferentialEvolution,
    "RO-V2": ResolutiveV2,
    "RO-V5": ResolutiveV5,
}


def run(*, budget: int, seeds: int, output: Path) -> None:
    if budget < 100:
        raise ValueError("budget must be >= 100")
    if seeds < 1:
        raise ValueError("seeds must be >= 1")

    rows = []
    for optimizer_name, optimizer_type in OPTIMIZERS.items():
        values = []
        evaluations = []
        solutions = []
        for seed in range(seeds):
            result = optimizer_type().minimize(
                pid_step_response_cost,
                dimension=3,
                bounds=(-1.0, 1.0),
                budget=budget,
                seed=seed,
            )
            values.append(float(result.fun))
            evaluations.append(int(result.evaluations))
            solutions.append(np.asarray(result.x, dtype=float))

        best_index = int(np.argmin(values))
        best_x = solutions[best_index]
        rows.append({
            "application": "pid_step_response",
            "optimizer": optimizer_name,
            "dimension": 3,
            "budget": budget,
            "seeds": seeds,
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if seeds > 1 else 0.0,
            "best": float(np.min(values)),
            "worst": float(np.max(values)),
            "best_x0": float(best_x[0]),
            "best_x1": float(best_x[1]),
            "best_x2": float(best_x[2]),
            "max_evaluations_used": int(np.max(evaluations)),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/pid_control.csv"))
    args = parser.parse_args()
    run(budget=args.budget, seeds=args.seeds, output=args.output)
