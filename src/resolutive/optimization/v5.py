"""Resolutive Optimizer V5 hybrid research engine.

RO-V5 keeps the strong RO-V2 single-funnel dynamics as its default behavior
and activates a short multi-funnel escape only after measured stagnation.
Individual mechanisms can be disabled for controlled ablation studies while
the default configuration preserves the full RO-V5 behavior.
"""

from __future__ import annotations

import numpy as np

from .common import Objective, OptimizationResult, validate_bounds


class ResolutiveV5:
    def __init__(
        self,
        population: int = 40,
        *,
        use_memory: bool = True,
        use_golden_rotation: bool = True,
        use_escape: bool = True,
        use_reanchor: bool = True,
        use_collapse: bool = True,
    ) -> None:
        if population < 12:
            raise ValueError("population must be >= 12")
        self.population = population
        self.use_memory = bool(use_memory)
        self.use_golden_rotation = bool(use_golden_rotation)
        self.use_escape = bool(use_escape)
        self.use_reanchor = bool(use_reanchor)
        self.use_collapse = bool(use_collapse)

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
        pop = self.population
        if budget <= pop:
            raise ValueError("budget must exceed population")

        rng = np.random.default_rng(seed)
        span = hi - lo
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        golden_angle = 2.0 * np.pi * (1.0 - 1.0 / phi)
        x = rng.uniform(lo, hi, (pop, dimension))
        values = np.array([objective(row) for row in x], dtype=float)
        used = pop
        best_i = int(np.argmin(values))
        best = x[best_i].copy()
        best_value = float(values[best_i])
        history = [best_value]
        bad_regions: list[tuple[np.ndarray, float, float]] = []
        stall = 0
        escape = False
        escape_left = 0
        generation = 0
        collapse_budget = max(300, budget // 15) if self.use_collapse else 0

        while used + pop <= budget - collapse_budget:
            generation += 1
            order = np.argsort(values)
            elite = x[order[: max(5, pop // 4)]]
            covariance = np.cov(elite, rowvar=False) + np.eye(dimension) * 1e-10
            eigvals, eigvecs = np.linalg.eigh(covariance)
            spread = np.sqrt(np.trace(covariance)) / span
            anisotropy = (eigvals[-1] + 1e-15) / (eigvals[0] + 1e-15)
            coherence = np.clip(
                0.5 * (used / budget) + 0.5 * (1.0 - np.clip(spread / 0.25, 0.0, 1.0)),
                0.0,
                1.0,
            )

            history = (history + [best_value])[-12:]
            relative_improvement = 0.0
            if len(history) >= 8:
                relative_improvement = (history[-8] - history[-1]) / (abs(history[-8]) + 1e-12)
            if self.use_escape and not escape and stall >= 10 and relative_improvement < 1e-5:
                escape = True
                escape_left = 8

            proposals = []
            if not escape:
                vertical = 0.28 * span * (1.0 - coherence) ** 2.2
                horizontal = 0.18 + 0.72 * coherence
                try:
                    chol = np.linalg.cholesky(covariance)
                except np.linalg.LinAlgError:
                    chol = np.eye(dimension) * max(np.std(elite), 1e-8)

                for i, row in enumerate(x):
                    target = elite[rng.integers(len(elite))]
                    direction = horizontal * (0.58 * (best - row) + 0.42 * (target - row))
                    if self.use_golden_rotation and dimension > 1:
                        a, b = rng.choice(dimension, 2, replace=False)
                        angle = golden_angle * (generation + i + 1)
                        va, vb = direction[a], direction[b]
                        direction[a] = np.cos(angle) * va - np.sin(angle) * vb
                        direction[b] = np.sin(angle) * va + np.cos(angle) * vb
                    repulsion = np.zeros(dimension)
                    if self.use_memory:
                        for center, radius, weight in bad_regions:
                            delta = row - center
                            repulsion += (
                                weight
                                * np.exp(-np.dot(delta, delta) / (2.0 * radius**2 + 1e-12))
                                * delta
                                / (radius**2 + 1e-12)
                            )
                    noise = chol @ rng.normal(size=dimension)
                    noise /= np.linalg.norm(noise) + 1e-12
                    proposals.append(
                        np.clip(row + direction + vertical * noise + 0.05 * repulsion, lo, hi)
                    )
            else:
                for ids in np.array_split(np.arange(pop), 4):
                    local_order = ids[np.argsort(values[ids])]
                    local_elite = x[local_order[: max(2, len(local_order) // 3)]]
                    local_best = local_elite[0]
                    for idx in ids:
                        row = x[idx]
                        target = local_elite[rng.integers(len(local_elite))]
                        noise = rng.normal(size=dimension)
                        noise /= np.linalg.norm(noise) + 1e-12
                        step = 0.24 * (local_best - row) + 0.20 * (target - row) + 0.18 * span * noise
                        if anisotropy > 80.0:
                            step += (
                                0.08
                                * span
                                * (1.0 if rng.random() < 0.5 else -1.0)
                                * eigvecs[:, -1]
                            )
                        proposals.append(np.clip(row + step, lo, hi))

            y = np.asarray(proposals)
            y_values = np.array([objective(row) for row in y], dtype=float)
            used += len(y)
            improved = y_values < values

            if self.use_memory:
                rejected = np.where(~improved)[0]
                if len(rejected) and generation % 4 == 0:
                    worst_rejected = rejected[np.argmax(y_values[rejected])]
                    radius = max(0.02 * span, 0.12 * span * (1.0 - coherence))
                    bad_regions.append((y[worst_rejected].copy(), radius, 1.0))
                    bad_regions = [
                        (center, radius, weight * 0.94)
                        for center, radius, weight in bad_regions
                        if weight * 0.94 > 0.15
                    ][-24:]

            x = np.where(improved[:, None], y, x)
            values = np.where(improved, y_values, values)
            current_i = int(np.argmin(values))
            if values[current_i] < best_value:
                best = x[current_i].copy()
                best_value = float(values[current_i])
                stall = 0
            else:
                stall += 1

            if escape:
                escape_left -= 1
                if escape_left <= 0:
                    escape = False
                    stall = 0
                    if self.use_reanchor:
                        worst = np.argsort(values)[-max(2, pop // 8):]
                        for idx in worst:
                            if used >= budget - collapse_budget:
                                break
                            x[idx] = np.clip(
                                best + rng.normal(0.0, 0.03 * span, dimension), lo, hi
                            )
                            values[idx] = float(objective(x[idx]))
                            used += 1

        if self.use_collapse:
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

        return OptimizationResult(best, best_value, used, seed, "RO-V5")
