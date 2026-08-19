"""Generalization benchmarks beyond the original axis-aligned test set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from resolutive.benchmarks.functions import DEFAULT_BENCHMARKS
from resolutive.benchmarks.transforms import GaussianNoisyObjective, shifted_rotated
from resolutive.optimization.baselines import DifferentialEvolution, RandomSearch, SimulatedAnnealing
from resolutive.optimization.hybrid_multires import ResolutiveHybridMultiResolution
from resolutive.optimization.hybrid_multires_adaptive import ResolutiveHybridMultiResolutionAdaptive
from resolutive.optimization.hybrid_multires_robust import ResolutiveHybridMultiResolutionRobust
from resolutive.optimization.hybrid_regime import ResolutiveHybridRegime
from resolutive.optimization.v2 import ResolutiveV2
from resolutive.optimization.v5 import ResolutiveV5
from resolutive.optimization.v6 import ResolutiveV6
from resolutive.optimization.v7 import ResolutiveV7

OPTIMIZERS = {
    "RandomSearch": RandomSearch,
    "SimulatedAnnealing": SimulatedAnnealing,
    "DifferentialEvolution": DifferentialEvolution,
    "RO-V2": ResolutiveV2,
    "RO-V5": ResolutiveV5,
    "RO-V6": ResolutiveV6,
    "RO-V7": ResolutiveV7,
    "RO-Hybrid-exp": ResolutiveHybridRegime,
    "RO-Hybrid-Multires-exp": ResolutiveHybridMultiResolution,
    "RO-Hybrid-Multires-Robust-exp": ResolutiveHybridMultiResolutionRobust,
    "RO-Hybrid-Multires-Adaptive-exp": ResolutiveHybridMultiResolutionAdaptive,
}


def run(*, dimension: int, budget: int, seeds: int, instance_seed: int,
        noise_fraction: float, output: Path) -> None:
    if dimension < 2:
        raise ValueError("dimension must be >= 2 for rotated generalization tests")
    if budget < 100:
        raise ValueError("budget must be >= 100")
    if seeds < 1:
        raise ValueError("seeds must be >= 1")
    if noise_fraction < 0.0:
        raise ValueError("noise_fraction must be >= 0")

    rows: list[dict[str, object]] = []
    for benchmark_name, (base_objective, bounds) in DEFAULT_BENCHMARKS.items():
        transformed, shift, _rotation = shifted_rotated(
            base_objective, dimension=dimension, bounds=bounds,
            seed=instance_seed, rotate=True,
        )
        reference_scale = max(1.0, abs(float(base_objective(np.full(dimension, 0.5)))))
        noise_sigma = noise_fraction * reference_scale
        cases = [
            ("shifted_rotated", budget, False),
            ("shifted_rotated_noisy", budget, True),
            ("short_budget", max(1000, budget // 10), False),
        ]
        for case_name, case_budget, noisy in cases:
            for optimizer_name, optimizer_type in OPTIMIZERS.items():
                values, evaluations = [], []
                for seed in range(seeds):
                    objective = transformed
                    if noisy:
                        objective = GaussianNoisyObjective(
                            transformed, sigma=noise_sigma,
                            seed=100_000 * instance_seed + seed,
                        )
                    result = optimizer_type().minimize(
                        objective, dimension=dimension, bounds=bounds,
                        budget=case_budget, seed=seed,
                    )
                    values.append(float(transformed(result.x)))
                    evaluations.append(int(result.evaluations))
                rows.append({
                    "benchmark": benchmark_name, "case": case_name,
                    "optimizer": optimizer_name, "dimension": dimension,
                    "budget": case_budget, "seeds": seeds,
                    "instance_seed": instance_seed,
                    "shift_norm": float(np.linalg.norm(shift)),
                    "noise_sigma": noise_sigma if noisy else 0.0,
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if seeds > 1 else 0.0,
                    "best": float(np.min(values)), "worst": float(np.max(values)),
                    "max_evaluations_used": int(np.max(evaluations)),
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--budget", type=int, default=6000)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--instance-seed", type=int, default=2026)
    parser.add_argument("--noise-fraction", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("results/generalization.csv"))
    args = parser.parse_args()
    run(dimension=args.dimension, budget=args.budget, seeds=args.seeds,
        instance_seed=args.instance_seed, noise_fraction=args.noise_fraction,
        output=args.output)
