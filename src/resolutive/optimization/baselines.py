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


class DifferentialEvolution:
    """Exact-budget DE/rand/1/bin reference implementation.

    The implementation evaluates one trial vector at a time after population
    initialization, so the reported budget is an exact objective-call count.
    """

    def __init__(
        self,
        *,
        population_multiplier: int = 8,
        mutation: float = 0.8,
        crossover: float = 0.9,
    ) -> None:
        if population_multiplier < 1:
            raise ValueError("population_multiplier must be >= 1")
        if not 0.0 < mutation <= 2.0:
            raise ValueError("mutation must be in (0, 2]")
        if not 0.0 <= crossover <= 1.0:
            raise ValueError("crossover must be in [0, 1]")
        self.population_multiplier = int(population_multiplier)
        self.mutation = float(mutation)
        self.crossover = float(crossover)

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
        lo, hi = validate_bounds(bounds)
        minimum_population = 4
        requested_population = max(minimum_population, self.population_multiplier * dimension)
        if budget < minimum_population:
            raise ValueError("budget must be >= 4 for differential evolution")
        population_size = min(requested_population, budget)

        rng = np.random.default_rng(seed)
        population = rng.uniform(lo, hi, (population_size, dimension))
        values = np.array([objective(row) for row in population], dtype=float)
        used = population_size
        target_index = 0

        while used < budget:
            available = np.delete(np.arange(population_size), target_index)
            if available.size < 3:
                raise RuntimeError("differential evolution requires at least four population members")
            a, b, c = rng.choice(available, 3, replace=False)
            mutant = population[a] + self.mutation * (population[b] - population[c])
            mutant = np.clip(mutant, lo, hi)

            crossover_mask = rng.random(dimension) < self.crossover
            crossover_mask[rng.integers(dimension)] = True
            trial = np.where(crossover_mask, mutant, population[target_index])
            trial_value = float(objective(trial))
            used += 1

            if trial_value <= values[target_index]:
                population[target_index] = trial
                values[target_index] = trial_value

            target_index = (target_index + 1) % population_size

        best_index = int(np.argmin(values))
        return OptimizationResult(
            population[best_index].copy(),
            float(values[best_index]),
            used,
            seed,
            "DifferentialEvolution",
        )
