"""Experimental RO-V2 + SpiralRelief hybrid."""
from __future__ import annotations

from .common import Objective, OptimizationResult, validate_bounds
from .spiral_relief import SpiralReliefOptimizer
from .v2 import ResolutiveV2


class ResolutiveV2Spiral:
    def __init__(self, population: int = 36, spiral_fraction: float = 0.15) -> None:
        if not 0.05 <= spiral_fraction <= 0.40:
            raise ValueError("spiral_fraction must be between 0.05 and 0.40")
        self.population = population
        self.spiral_fraction = float(spiral_fraction)

    def minimize(self, objective: Objective, *, dimension: int,
                 bounds: tuple[float, float], budget: int = 6000,
                 seed: int = 0) -> OptimizationResult:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        lo, hi = validate_bounds(bounds)
        reserve = max(120, int(round(budget * self.spiral_fraction)))
        global_budget = budget - reserve
        if global_budget <= self.population + 50:
            raise ValueError("budget too small for hybrid optimization")

        base = ResolutiveV2(population=self.population).minimize(
            objective, dimension=dimension, bounds=(lo, hi),
            budget=global_budget, seed=seed,
        )
        remaining = max(0, budget - base.evaluations)
        if remaining < 20:
            return OptimizationResult(base.x, base.fun, base.evaluations, seed, "RO-V2-Spiral-exp")

        refined = SpiralReliefOptimizer(spiral_points=18, turns=2.5).refine(
            objective, start=base.x, bounds=(lo, hi), budget=remaining,
            seed=seed + 200003, radius_fraction=0.035,
        )
        if refined.fun < base.fun:
            x, fun = refined.x, refined.fun
        else:
            x, fun = base.x, base.fun
        return OptimizationResult(
            x, float(fun), base.evaluations + refined.evaluations,
            seed, "RO-V2-Spiral-exp",
        )
