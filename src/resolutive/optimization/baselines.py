"""Budget-matched classical baselines for continuous black-box optimization.

These implementations deliberately share the same minimal contract as the
Resolutive Optimizers: objective, dimension, scalar bounds, evaluation budget,
and seed. They are reference baselines, not claims of state-of-the-art tuning.
"""

from __future__ import annotations

import numpy as np

from .common import Objective, OptimizationResult, validate_bounds


class RandomSearch:
    """Uniform random search with an exact evaluation budget."""

    def minimize(
        self,
        objective: Objective,
        *,
        dimension: int,
        bounds: tuple[float, float],
        budget: int = 6000,
        seed: int = 0,
    ) -> OptimizationResult:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        if budget < 1:
            raise ValueError("budget must be >= 1")
        lo, hi = validate_bounds(bounds)
        rng = np.random.default_rng(seed)
        best = None
        best_value = np.inf
        for _ in range(budget):
            candidate = rng.uniform(lo, hi, dimension)
            value = float(objective(candidate))
            if value < best_value:
                best = candidate.copy()
                best_value = value
        assert best is not None
        return OptimizationResult(best, best_value, budget, seed, "RandomSearch")


class SimulatedAnnealing:
    """Simple bounded simulated annealing reference implementation."""

    def __init__(self, initial_step_fraction: float = 0.15) -> None:
        if not 0.0 < initial_step_fraction <= 1.0:
            raise ValueError("initial_step_fraction must be in (0, 1]")
        self.initial_step_fraction = float(initial_step_fraction)

    def minimize(
        self,
        objective: Objective,
        *,
        dimension: int,
        bounds: tuple[float, float],
        budget: int = 6000,
        seed: int = 0,
    ) -> OptimizationResult:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        if budget < 2:
            raise ValueError("budget must be >= 2")
        lo, hi = validate_bounds(bounds)
        span = hi - lo
        rng = np.random.default_rng(seed)
        current = rng.uniform(lo, hi, dimension)
        current_value = float(objective(current))
        best = current.copy()
        best_value = current_value
        used = 1

        while used < budget:
            progress = used / budget
            temperature = max(1e-6, 1.0 - progress)
            step = self.initial_step_fraction * span * temperature**0.7
            candidate = np.clip(current + rng.normal(0.0, step, dimension), lo, hi)
            value = float(objective(candidate))
            used += 1

            scale = temperature * max(1.0, abs(current_value)) + 1e-12
            accept = value < current_value
            if not accept:
                accept = rng.random() < np.exp(-(value - current_value) / scale)
            if accept:
                current = candidate
                current_value = value
                if value < best_value:
                    best = candidate.copy()
                    best_value = value

        return OptimizationResult(best, best_value, used, seed, "SimulatedAnnealing")
