"""Experimental RO-V5 + local SpiralRelief hybrid.

RO-V5 performs the global search. A small fraction of the total objective
budget is reserved for local map-and-slide refinement around the V5 incumbent.
The implementation remains separate from RO-V5 until benchmarks justify it.
"""
from __future__ import annotations

from .common import Objective, OptimizationResult, validate_bounds
from .spiral_relief import SpiralReliefOptimizer
from .v5 import ResolutiveV5


class ResolutiveV5Spiral:
    def __init__(self, spiral_fraction: float = 0.15) -> None:
        if not (0.05 <= spiral_fraction <= 0.35):
            raise ValueError("spiral_fraction must be in [0.05, 0.35]")
        self.spiral_fraction = float(spiral_fraction)

    def minimize(self, objective: Objective, *, dimension: int,
                 bounds: tuple[float, float], budget: int = 6000,
                 seed: int = 0) -> OptimizationResult:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        validate_bounds(bounds)
        if budget < 500:
            raise ValueError("budget must be >= 500")

        local_budget = max(120, int(round(budget * self.spiral_fraction)))
        global_budget = budget - local_budget
        # Disable V5 collapse: the reserved local SpiralRelief replaces the final
        # coordinate-only collapse while preserving V5's global dynamics.
        global_result = ResolutiveV5(use_collapse=False).minimize(
            objective, dimension=dimension, bounds=bounds,
            budget=global_budget, seed=seed,
        )

        local = SpiralReliefOptimizer(spiral_points=18, turns=2.5).refine(
            objective, start=global_result.x, bounds=bounds,
            budget=local_budget, seed=100_003 + seed,
            radius_fraction=0.035,
        )
        if local.fun < global_result.fun:
            x, fun = local.x, local.fun
        else:
            x, fun = global_result.x, global_result.fun
        used = global_result.evaluations + local.evaluations
        return OptimizationResult(x, float(fun), used, seed, "RO-V5-Spiral-exp")
