"""Resolutive Optimizer V7: hybrid V5 core with gated geometry mode.

V7 keeps the stable V5 search dynamics as the default regime. A covariance-
aware geometry mode is activated only when the population is both stagnant and
strongly anisotropic, then automatically returns to the V5-like regime after a
short burst. This is intended to preserve V5's multimodal performance while
using learned directions only where axis bias is likely to matter.
"""
from __future__ import annotations

import numpy as np

from .common import Objective, OptimizationResult, validate_bounds


class ResolutiveV7:
    def __init__(self, population: int = 40) -> None:
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
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        golden_angle = 2.0 * np.pi * (1.0 - 1.0 / phi)

        x = rng.uniform(lo, hi, (pop, dimension))
        values = np.array([objective(v) for v in x], dtype=float)
        used = pop
        best_i = int(np.argmin(values))
        best = x[best_i].copy()
        best_value = float(values[best_i])

        stall = 0
        generation = 0
        geometry_left = 0
        bad_regions: list[tuple[np.ndarray, float, float]] = []
        collapse_budget = max(300, budget // 15)

        while used + pop <= budget - collapse_budget:
            generation += 1
            order = np.argsort(values)
            elite = x[order[: max(5, pop // 4)]]
            cov = np.cov(elite, rowvar=False) + np.eye(dimension) * 1e-10
            eigvals, eigvecs = np.linalg.eigh(cov)
            spread = np.sqrt(np.trace(cov)) / span
            anisotropy = (eigvals[-1] + 1e-15) / (eigvals[0] + 1e-15)
            coherence = float(np.clip(
                0.5 * (used / budget) + 0.5 * (1.0 - np.clip(spread / 0.25, 0.0, 1.0)),
                0.0, 1.0,
            ))

            if geometry_left == 0 and stall >= 8 and anisotropy > 40.0:
                geometry_left = 5

            proposals = []
            if geometry_left == 0:
                # V5-like stable regime.
                vertical = 0.28 * span * (1.0 - coherence) ** 2.2
                horizontal = 0.18 + 0.72 * coherence
                try:
                    chol = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    chol = np.eye(dimension) * max(np.std(elite), 1e-8)

                for i, row in enumerate(x):
                    target = elite[rng.integers(len(elite))]
                    direction = horizontal * (0.58 * (best - row) + 0.42 * (target - row))
                    if dimension > 1:
                        a, b = rng.choice(dimension, 2, replace=False)
                        angle = golden_angle * (generation + i + 1)
                        va, vb = direction[a], direction[b]
                        direction[a] = np.cos(angle) * va - np.sin(angle) * vb
                        direction[b] = np.sin(angle) * va + np.cos(angle) * vb

                    repulsion = np.zeros(dimension)
                    for center, radius, weight in bad_regions:
                        delta = row - center
                        repulsion += (
                            weight
                            * np.exp(-np.dot(delta, delta) / (2.0 * radius**2 + 1e-12))
                            * delta / (radius**2 + 1e-12)
                        )
                    noise = chol @ rng.normal(size=dimension)
                    noise /= np.linalg.norm(noise) + 1e-12
                    proposals.append(np.clip(
                        row + direction + vertical * noise + 0.05 * repulsion,
                        lo, hi,
                    ))
            else:
                # Short geometry-aware burst, used only for anisotropic stagnation.
                scales = np.sqrt(np.maximum(eigvals, 1e-14))
                scales /= np.sqrt(np.mean(np.maximum(eigvals, 1e-14))) + 1e-15
                transform = eigvecs @ np.diag(scales) @ eigvecs.T
                step_scale = 0.10 * span * (0.5 + 0.5 * (1.0 - coherence))
                for row in x:
                    target = elite[rng.integers(len(elite))]
                    z = rng.normal(size=dimension)
                    z /= np.linalg.norm(z) + 1e-15
                    proposal = (
                        row
                        + 0.42 * (best - row)
                        + 0.24 * (target - row)
                        + step_scale * (transform @ z)
                    )
                    proposals.append(np.clip(proposal, lo, hi))

            y = np.asarray(proposals)
            y_values = np.array([objective(v) for v in y], dtype=float)
            used += len(y)
            improved = y_values < values

            rejected = np.where(~improved)[0]
            if len(rejected) and generation % 4 == 0:
                j = rejected[int(np.argmax(y_values[rejected]))]
                radius = max(0.02 * span, 0.12 * span * (1.0 - coherence))
                bad_regions.append((y[j].copy(), radius, 1.0))
                bad_regions = [
                    (c, r, w * 0.94) for c, r, w in bad_regions if w * 0.94 > 0.15
                ][-24:]

            x = np.where(improved[:, None], y, x)
            values = np.where(improved, y_values, values)
            current = int(np.argmin(values))
            if float(values[current]) < best_value:
                best = x[current].copy()
                best_value = float(values[current])
                stall = 0
            else:
                stall += 1

            if geometry_left > 0:
                geometry_left -= 1
                if geometry_left == 0:
                    stall = 0
                    # Reanchor only the weakest fraction near the current best.
                    worst = np.argsort(values)[-max(2, pop // 10):]
                    for idx in worst:
                        if used >= budget - collapse_budget:
                            break
                        candidate = np.clip(best + rng.normal(0.0, 0.025 * span, dimension), lo, hi)
                        x[idx] = candidate
                        values[idx] = float(objective(candidate))
                        used += 1

        # Preserve V5 coordinate collapse: it was empirically more stable than V6's
        # principal-direction collapse in the first strong-baseline campaign.
        point = best.copy()
        step = 0.05 * span
        while used + 2 * dimension <= budget and step > 1e-10 * span:
            improved_any = False
            for axis in range(dimension):
                for sign in (-1.0, 1.0):
                    candidate = point.copy()
                    candidate[axis] = np.clip(candidate[axis] + sign * step, lo, hi)
                    val = float(objective(candidate))
                    used += 1
                    if val < best_value:
                        point = candidate
                        best = candidate.copy()
                        best_value = val
                        improved_any = True
            if not improved_any:
                step *= 0.5

        return OptimizationResult(best, best_value, used, seed, "RO-V7")
