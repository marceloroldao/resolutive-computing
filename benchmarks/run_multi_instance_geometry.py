"""Multi-instance geometry diagnostic for RO-V5/V6/V7 and CMA-ES.

Purpose: test whether apparent gains on shifted/rotated problems persist across
multiple deterministic problem instances instead of depending on one rotation.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cma
import numpy as np

from resolutive.benchmarks.functions import DEFAULT_BENCHMARKS
from resolutive.benchmarks.transforms import GaussianNoisyObjective, shifted_rotated
from resolutive.optimization.v5 import ResolutiveV5
from resolutive.optimization.v6 import ResolutiveV6
from resolutive.optimization.v7 import ResolutiveV7


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
            best = float(ys[j]); best_x = np.asarray(xs[j], dtype=float).copy()
        es.tell(xs, ys)
    return best_x, best, used


def run(*, dimension: int, budget: int, seeds: int, instance_seeds: list[int],
        noise_fraction: float, output: Path) -> None:
    if dimension < 2:
        raise ValueError("dimension must be >= 2")
    if budget < 100:
        raise ValueError("budget must be >= 100")
    if seeds < 1 or not instance_seeds:
        raise ValueError("seeds and instance_seeds must be non-empty")

    optimizers = {
        "CMA-ES(pycma)": None,
        "RO-V5": ResolutiveV5,
        "RO-V6": ResolutiveV6,
        "RO-V7": ResolutiveV7,
    }
    rows: list[dict[str, object]] = []

    for benchmark_name, (base_objective, bounds) in DEFAULT_BENCHMARKS.items():
        reference_scale = max(1.0, abs(float(base_objective(np.full(dimension, 0.5)))))
        noise_sigma = noise_fraction * reference_scale
        for instance_seed in instance_seeds:
            transformed, shift, _rotation = shifted_rotated(
                base_objective, dimension=dimension, bounds=bounds,
                seed=instance_seed, rotate=True,
            )
            for case_name, noisy in (("shifted_rotated", False), ("shifted_rotated_noisy", True)):
                for optimizer_name, optimizer_type in optimizers.items():
                    values, evals = [], []
                    for seed in range(seeds):
                        objective = transformed
                        if noisy:
                            objective = GaussianNoisyObjective(
                                transformed, sigma=noise_sigma,
                                seed=100_000 * instance_seed + seed,
                            )
                        if optimizer_name.startswith("CMA-ES"):
                            x, _observed, used = _cma_es(objective, dimension, bounds, budget, seed)
                        else:
                            result = optimizer_type().minimize(
                                objective, dimension=dimension, bounds=bounds,
                                budget=budget, seed=seed,
                            )
                            x, used = result.x, int(result.evaluations)
                        values.append(float(transformed(x)))
                        evals.append(used)
                    rows.append({
                        "benchmark": benchmark_name,
                        "case": case_name,
                        "instance_seed": instance_seed,
                        "optimizer": optimizer_name,
                        "dimension": dimension,
                        "budget": budget,
                        "seeds": seeds,
                        "shift_norm": float(np.linalg.norm(shift)),
                        "noise_sigma": noise_sigma if noisy else 0.0,
                        "median": float(np.median(values)),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if seeds > 1 else 0.0,
                        "best": float(np.min(values)),
                        "worst": float(np.max(values)),
                        "max_evaluations_used": int(np.max(evals)),
                    })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dimension", type=int, default=10)
    p.add_argument("--budget", type=int, default=3000)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--instance-seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    p.add_argument("--noise-fraction", type=float, default=0.01)
    p.add_argument("--output", type=Path, default=Path("results/multi_instance_geometry.csv"))
    a = p.parse_args()
    run(dimension=a.dimension, budget=a.budget, seeds=a.seeds,
        instance_seeds=a.instance_seeds, noise_fraction=a.noise_fraction,
        output=a.output)
