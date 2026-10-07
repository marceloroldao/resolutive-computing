"""Resolutive Optimizer V6: geometry-adaptive experimental engine.

V6 preserves the resolutive ideas of coherence, memory, escape and collapse,
while removing coordinate-axis bias. Its main additions are:
- covariance-eigenbasis transport instead of pairwise coordinate rotation;
- success-rate adaptation of exploration amplitude;
- diversity-triggered partial restart;
- final collapse along learned principal directions.

This is experimental and must be evaluated out-of-sample before promotion.
"""
from __future__ import annotations

import numpy as np

from .common import Objective, OptimizationResult, validate_bounds


class ResolutiveV6:
    def __init__(self, population: int = 48) -> None:
        if population < 16:
            raise ValueError("population must be >= 16")
        self.population = population

    def minimize(self, objective: Objective, *, dimension: int,
                 bounds: tuple[float, float], budget: int = 6000,
                 seed: int = 0) -> OptimizationResult:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        lo, hi = validate_bounds(bounds)
        pop = self.population
        if budget <= pop:
            raise ValueError("budget must exceed population")

        rng = np.random.default_rng(seed)
        span = hi - lo
        x = rng.uniform(lo, hi, (pop, dimension))
        values = np.array([objective(v) for v in x], dtype=float)
        used = pop
        best_i = int(np.argmin(values))
        best = x[best_i].copy(); best_value = float(values[best_i])

        sigma = 0.16 * span
        success_ema = 0.20
        stall = 0
        memory: list[tuple[np.ndarray, float]] = []
        collapse_budget = max(400, budget // 12)

        while used + pop <= budget - collapse_budget:
            order = np.argsort(values)
            elite = x[order[: max(6, pop // 4)]]
            center = np.mean(elite, axis=0)
            cov = np.cov(elite, rowvar=False) + np.eye(dimension) * (1e-12 * span * span)
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-14 * span * span)

            # Learned geometry: normalize eigen-scales so sigma remains interpretable.
            scales = np.sqrt(eigvals)
            scales /= np.sqrt(np.mean(eigvals)) + 1e-15
            transform = eigvecs @ np.diag(scales) @ eigvecs.T

            diversity = float(np.sqrt(np.trace(cov)) / (span * np.sqrt(dimension)))
            progress = used / budget
            coherence = float(np.clip(0.45 * progress + 0.55 * (1.0 - min(diversity / 0.18, 1.0)), 0, 1))

            proposals = []
            for i, row in enumerate(x):
                target = elite[rng.integers(len(elite))]
                attract = (0.36 + 0.36 * coherence) * (best - row) + 0.24 * (target - row)
                # Whole-space geometry transport; rotation-invariant under orthogonal transforms.
                z = rng.normal(size=dimension)
                z /= np.linalg.norm(z) + 1e-15
                explore = sigma * (0.35 + 0.65 * (1.0 - coherence)) * (transform @ z)

                repel = np.zeros(dimension)
                for c, w in memory:
                    d = row - c
                    d2 = float(np.dot(d, d)) + 1e-12
                    repel += w * np.exp(-d2 / (0.08 * span * span)) * d / np.sqrt(d2)

                # Occasional differential displacement helps cross multimodal basins.
                a, b = rng.choice(pop, 2, replace=False)
                differential = 0.12 * (1.0 - coherence) * (x[a] - x[b])
                proposal = row + attract + explore + differential + 0.025 * span * repel
                proposals.append(np.clip(proposal, lo, hi))

            y = np.asarray(proposals)
            y_values = np.array([objective(v) for v in y], dtype=float)
            used += pop
            improved = y_values < values
            success = float(np.mean(improved))
            success_ema = 0.8 * success_ema + 0.2 * success

            # General-purpose 1/5-success adaptation, bounded to avoid collapse/explosion.
            if success_ema > 0.24:
                sigma *= 1.08
            elif success_ema < 0.16:
                sigma *= 0.90
            sigma = float(np.clip(sigma, 1e-5 * span, 0.35 * span))

            rejected = np.where(~improved)[0]
            if len(rejected):
                j = rejected[int(np.argmax(y_values[rejected]))]
                memory.append((y[j].copy(), 1.0))
                memory = [(c, w * 0.92) for c, w in memory if w * 0.92 > 0.18][-20:]

            x = np.where(improved[:, None], y, x)
            values = np.where(improved, y_values, values)
            current = int(np.argmin(values))
            if float(values[current]) < best_value:
                best = x[current].copy(); best_value = float(values[current]); stall = 0
            else:
                stall += 1

            # Diversity/stagnation restart: retain elite, reseed only the weakest fraction.
            if stall >= 9 and (diversity < 0.025 or success_ema < 0.08):
                worst = np.argsort(values)[-max(3, pop // 5):]
                for idx in worst:
                    if used >= budget - collapse_budget:
                        break
                    # mixture of near-best and global reseeding
                    if rng.random() < 0.65:
                        candidate = best + rng.normal(0, 0.12 * span, dimension)
                    else:
                        candidate = rng.uniform(lo, hi, dimension)
                    candidate = np.clip(candidate, lo, hi)
                    x[idx] = candidate
                    values[idx] = float(objective(candidate)); used += 1
                stall = 0
                sigma = max(sigma, 0.08 * span)

        # Rotation-aware principal-direction collapse instead of coordinate-axis search.
        order = np.argsort(values)
        elite = x[order[: max(6, pop // 4)]]
        cov = np.cov(elite, rowvar=False) + np.eye(dimension) * 1e-12
        _, basis = np.linalg.eigh(cov)
        point = best.copy(); step = 0.05 * span
        while used + 2 * dimension <= budget and step > 1e-10 * span:
            improved_any = False
            # Search learned directions from largest to smallest variance.
            for axis in range(dimension - 1, -1, -1):
                direction = basis[:, axis]
                for sign in (-1.0, 1.0):
                    candidate = np.clip(point + sign * step * direction, lo, hi)
                    val = float(objective(candidate)); used += 1
                    if val < best_value:
                        point = candidate; best = candidate.copy(); best_value = val
                        improved_any = True
            if not improved_any:
                step *= 0.5

        return OptimizationResult(best, best_value, used, seed, "RO-V6")
