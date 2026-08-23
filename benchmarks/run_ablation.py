"""Controlled RO-V5 ablation study.

Each row disables exactly one RO-V5 mechanism while preserving the same
benchmark, evaluation budget, dimension and seed set used by the full model.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from resolutive.benchmarks.functions import DEFAULT_BENCHMARKS
from resolutive.optimization.v5 import ResolutiveV5


VARIANTS = {
    "RO-V5-full": {},
    "RO-V5-no-memory": {"use_memory": False},
    "RO-V5-no-golden-rotation": {"use_golden_rotation": False},
    "RO-V5-no-escape": {"use_escape": False},
    "RO-V5-no-reanchor": {"use_reanchor": False},
    "RO-V5-no-collapse": {"use_collapse": False},
}


def run(dimension: int, budget: int, seeds: int, output: Path) -> None:
    rows = []
    for benchmark_name, (objective, bounds) in DEFAULT_BENCHMARKS.items():
        for variant_name, options in VARIANTS.items():
            optimizer = ResolutiveV5(**options)
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
                    "variant": variant_name,
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
    parser.add_argument("--output", type=Path, default=Path("results/ablation_v5.csv"))
    args = parser.parse_args()
    run(args.dimension, args.budget, args.seeds, args.output)
