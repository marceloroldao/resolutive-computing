"""Experimental multiresolution extension of the regime-selecting hybrid.

The existing RO-Hybrid is preserved unchanged. This variant spends most of the
budget on RO-Hybrid, then repeatedly probes a local spiral relief at shrinking
radii. Each level accepts only objectively verified improvements.
"""
from __future__ import annotations

import numpy as np

from .common import Objective, OptimizationResult, validate_bounds
from .hybrid_regime import ResolutiveHybridRegime
from .spiral_relief import _fit_relief, _orthonormal_plane


class ResolutiveHybridMultiResolution:
    def __init__(
        self,
        local_fraction: float = 0.20,
        radius_schedule: tuple[float, ...] = (0.04, 0.01, 0.0025, 0.000625),
        points_per_level: int = 18,
        turns: float = 2.5,
    ) -> None:
        if not 0.12 <= local_fraction <= 0.35:
            raise ValueError("local_fraction must be in [0.12, 0.35]")
        if len(radius_schedule) < 2 or any(r <= 0 for r in radius_schedule):
            raise ValueError("radius_schedule must contain at least two positive radii")
        if any(b >= a for a, b in zip(radius_schedule, radius_schedule[1:])):
            raise ValueError("radius_schedule must be strictly decreasing")
        if points_per_level < 8:
            raise ValueError("points_per_level must be >= 8")
        self.local_fraction = float(local_fraction)
        self.radius_schedule = tuple(float(r) for r in radius_schedule)
        self.points_per_level = int(points_per_level)
        self.turns = float(turns)

    def minimize(
        self,
        objective: Objective,
        *,
        dimension: int,
        bounds: tuple[float, float],
        budget: int = 6000,
        seed: int = 0,
    ) -> OptimizationResult:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        lo, hi = validate_bounds(bounds)
        if budget < 800:
            raise ValueError("budget must be >= 800")

        local_budget = max(220, int(round(budget * self.local_fraction)))
        core_budget = budget - local_budget
        core = ResolutiveHybridRegime(local_fraction=0.10).minimize(
            objective,
            dimension=dimension,
            bounds=(lo, hi),
            budget=core_budget,
            seed=seed,
        )
        best_x = core.x.copy()
        best_f = float(core.fun)
        used = int(core.evaluations)
        span = hi - lo
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        # Progressive resolution: broad local map -> progressively finer maps.
        for level, radius_fraction in enumerate(self.radius_schedule):
            remaining = budget - used
            levels_left = len(self.radius_schedule) - level
            if remaining < 10:
                break
            level_budget = max(8, remaining // levels_left)
            n = min(self.points_per_level, max(8, level_budget - 2))
            if n + 1 > remaining:
                n = remaining - 1
            if n < 8:
                break

            rng = np.random.default_rng(seed + 810001 + level * 9973)
            u, v = _orthonormal_plane(rng, dimension)
            idx = np.arange(1, n + 1, dtype=float)
            radial = radius_fraction * span * np.sqrt(idx / n)
            theta = golden_angle * idx * self.turns
            coords = np.column_stack((radial * np.cos(theta), radial * np.sin(theta)))
            points = np.clip(best_x + coords[:, :1] * u + coords[:, 1:] * v, lo, hi)
            vals = np.array([float(objective(p)) for p in points])
            used += n

            j = int(np.argmin(vals))
            if vals[j] < best_f:
                best_x = points[j].copy()
                best_f = float(vals[j])

            # Reconstruct the local relief and attempt one geometry-derived slide.
            if used < budget:
                grad, hessian = _fit_relief(coords, vals)
                eig = np.linalg.eigvalsh(hessian)
                step2 = None
                if np.all(eig > 1e-10):
                    try:
                        step2 = -np.linalg.solve(hessian + 1e-10 * np.eye(2), grad)
                    except np.linalg.LinAlgError:
                        step2 = None
                if step2 is None or not np.all(np.isfinite(step2)):
                    norm = float(np.linalg.norm(grad))
                    if norm > 1e-15:
                        step2 = -(radius_fraction * span) * grad / norm
                if step2 is not None and np.all(np.isfinite(step2)):
                    max_step = radius_fraction * span
                    norm = float(np.linalg.norm(step2))
                    if norm > max_step:
                        step2 = step2 * (max_step / (norm + 1e-15))
                    cand = np.clip(best_x + step2[0] * u + step2[1] * v, lo, hi)
                    val = float(objective(cand))
                    used += 1
                    if val < best_f:
                        best_x, best_f = cand.copy(), val

        # Use any tiny remainder for a resolution-matched coordinate polish.
        if used + 2 * dimension <= budget:
            step = self.radius_schedule[-1] * span
            while used + 2 * dimension <= budget and step > 1e-13 * span:
                improved = False
                for axis in range(dimension):
                    for sign in (-1.0, 1.0):
                        cand = best_x.copy()
                        cand[axis] = np.clip(cand[axis] + sign * step, lo, hi)
                        val = float(objective(cand))
                        used += 1
                        if val < best_f:
                            best_x, best_f = cand, val
                            improved = True
                if not improved:
                    step *= 0.25

        return OptimizationResult(
            best_x,
            best_f,
            used,
            seed,
            "RO-Hybrid-Multires-exp",
        )
