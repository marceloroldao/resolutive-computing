"""Resolutive Optimizer V2.

This module preserves the strong single-funnel baseline used in the early
benchmark program: coherence-controlled exploration, elite covariance and a
coordinate-collapse refinement stage.
"""

from __future__ import annotations

import numpy as np

from .common import Objective, OptimizationResult, validate_bounds


class ResolutiveV2:
    def __init__(self, population: int = 36) -> None:
        if population < 8:
            raise ValueError("population must be >= 8")
        self.population = population

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
        if budget <= self.population:
            raise ValueError("budget must exceed population")

        rng = np.random.default_rng(seed)
        span = hi - lo
        pop = self.population
        x = rng.uniform(lo, hi, (pop, dimension))
        values = np.array([objective(row) for row in x], dtype=float)
        used = pop
        best_i = int(np.argmin(values))
        best = x[best_i].copy()
        best_value = float(values[best_i])
        stall = 0
        generation = 0
        golden_angle = 2.0 * np.pi * (1.0 - 1.0 / ((1.0 + np.sqrt(5.0)) / 2.0))
        collapse_budget = max(200, budget // 20)

        while used + pop <= budget - collapse_budget:
            generation += 1
            order = np.argsort(values)
            elite = x[order[: max(5, pop // 4)]]
            covariance = np.cov(elite, rowvar=False) + np.eye(dimension) * 1e-10
            spread = np.sqrt(np.trace(covariance)) / span
            progress = used / budget
            coherence = np.clip(
                0.5 * progress + 0.5 * (1.0 - np.clip(spread / 0.25, 0.0, 1.0)),
                0.0,
                1.0,
            )
            vertical = 0.28 * span * (1.0 - coherence) ** 2.2
            horizontal = 0.18 + 0.72 * coherence

            proposals = []
            for i, row in enumerate(x):
                target = elite[rng.integers(len(elite))]
                direction = horizontal * (0.58 * (best - row) + 0.42 * (target - row))
                if dimension > 1:
                    a, b = rng.choice(dimension, 2, replace=False)
                    angle = golden_angle * (generation + i + 1)
                    va, vb = direction[a], direction[b]
                    direction[a] = np.cos(angle) * va - np.sin(angle) * vb
                    direction[b] = np.sin(angle) * va + np.cos(angle) * vb
                noise = rng.multivariate_normal(np.zeros(dimension), covariance)
                noise /= np.linalg.norm(noise) + 1e-12
                proposals.append(np.clip(row + direction + vertical * noise, lo, hi))

            y = np.asarray(proposals)
            y_values = np.array([objective(row) for row in y], dtype=float)
            used += pop
            improved = y_values < values
            x = np.where(improved[:, None], y, x)
            values = np.where(improved, y_values, values)
            current_i = int(np.argmin(values))
            if values[current_i] < best_value:
                best = x[current_i].copy()
                best_value = float(values[current_i])
                stall = 0
            else:
                stall += 1
            if stall >= 14 and used < budget - collapse_budget:
                worst = np.argsort(values)[-max(2, pop // 8):]
                x[worst] = rng.uniform(lo, hi, (len(worst), dimension))
                values[worst] = np.array([objective(row) for row in x[worst]], dtype=float)
                used += len(worst)
                stall = 0

        point = best.copy()
        step = 0.05 * span
        while used + 2 * dimension <= budget and step > 1e-10 * span:
            improved_any = False
            for axis in range(dimension):
                for sign in (-1.0, 1.0):
                    candidate = point.copy()
                    candidate[axis] = np.clip(candidate[axis] + sign * step, lo, hi)
                    value = float(objective(candidate))
                    used += 1
                    if value < best_value:
                        point = candidate
                        best = candidate.copy()
                        best_value = value
                        improved_any = True
            if not improved_any:
                step *= 0.5

        return OptimizationResult(best, best_value, used, seed, "RO-V2")
