"""Experimental robust multiresolution optimizer.

This variant adapts the TRIVAX robust-observation idea to black-box optimization:
- local median re-evaluation of suspicious/promising points;
- MAD-based clipping of one-sample shocks before fitting the local relief;
- repeated confirmation of geometry-derived candidate moves.

The existing RO-Hybrid-Multires implementation is intentionally left unchanged
so the effect of robust evaluation can be measured by ablation.
"""
from __future__ import annotations

import numpy as np

from .common import Objective, OptimizationResult, validate_bounds
from .hybrid_regime import ResolutiveHybridRegime
from .spiral_relief import _fit_relief, _orthonormal_plane


def _mad_clip(values: np.ndarray, clip_scale: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if not np.isfinite(mad) or mad <= 1e-15:
        return values.copy()
    sigma = 1.4826 * mad
    limit = clip_scale * sigma
    return med + np.clip(values - med, -limit, limit)


class ResolutiveHybridMultiResolutionRobust:
    """RO-Hybrid-Multires with budget-aware robust local observations."""

    def __init__(
        self,
        local_fraction: float = 0.28,
        radius_schedule: tuple[float, ...] = (0.04, 0.01, 0.0025, 0.000625),
        points_per_level: int = 15,
        turns: float = 2.5,
        repeats: int = 3,
        elite_resamples: int = 3,
        clip_scale: float = 3.0,
    ) -> None:
        if not 0.20 <= local_fraction <= 0.40:
            raise ValueError("local_fraction must be in [0.20, 0.40]")
        if len(radius_schedule) < 2 or any(r <= 0 for r in radius_schedule):
            raise ValueError("radius_schedule must contain at least two positive radii")
        if any(b >= a for a, b in zip(radius_schedule, radius_schedule[1:])):
            raise ValueError("radius_schedule must be strictly decreasing")
        if points_per_level < 8:
            raise ValueError("points_per_level must be >= 8")
        if repeats < 3 or repeats % 2 == 0:
            raise ValueError("repeats must be an odd integer >= 3")
        if elite_resamples < 1:
            raise ValueError("elite_resamples must be >= 1")
        if clip_scale <= 0:
            raise ValueError("clip_scale must be positive")
        self.local_fraction = float(local_fraction)
        self.radius_schedule = tuple(float(r) for r in radius_schedule)
        self.points_per_level = int(points_per_level)
        self.turns = float(turns)
        self.repeats = int(repeats)
        self.elite_resamples = int(elite_resamples)
        self.clip_scale = float(clip_scale)

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
        if budget < 1000:
            raise ValueError("budget must be >= 1000")

        local_budget = max(360, int(round(budget * self.local_fraction)))
        core_budget = budget - local_budget
        if core_budget < 600:
            core_budget = 600
            local_budget = budget - core_budget

        core = ResolutiveHybridRegime(local_fraction=0.10).minimize(
            objective,
            dimension=dimension,
            bounds=(lo, hi),
            budget=core_budget,
            seed=seed,
        )
        best_x = core.x.copy()
        used = int(core.evaluations)

        # TRIVAX-inspired repeated confirmation: do not trust one noisy best value.
        confirmation = [float(core.fun)]
        for _ in range(self.repeats - 1):
            if used >= budget:
                break
            confirmation.append(float(objective(best_x)))
            used += 1
        best_f = float(np.median(confirmation))

        span = hi - lo
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        for level, radius_fraction in enumerate(self.radius_schedule):
            remaining = budget - used
            levels_left = len(self.radius_schedule) - level
            if remaining < 12:
                break

            # Reserve repeated confirmation evaluations inside each level.
            level_budget = max(10, remaining // levels_left)
            reserve = self.elite_resamples * (self.repeats - 1) + self.repeats
            n = min(self.points_per_level, max(8, level_budget - reserve))
            if n < 8 or used + n > budget:
                break

            rng = np.random.default_rng(seed + 910001 + level * 9973)
            u, v = _orthonormal_plane(rng, dimension)
            idx = np.arange(1, n + 1, dtype=float)
            radial = radius_fraction * span * np.sqrt(idx / n)
            theta = golden_angle * idx * self.turns
            coords = np.column_stack((radial * np.cos(theta), radial * np.sin(theta)))
            points = np.clip(best_x + coords[:, :1] * u + coords[:, 1:] * v, lo, hi)

            raw = np.array([float(objective(p)) for p in points], dtype=float)
            used += n
            robust = _mad_clip(raw, self.clip_scale)

            # Re-sample only the most promising points. This preserves most of the
            # budget while suppressing lucky one-sample minima caused by noise.
            elite_count = min(self.elite_resamples, n)
            elite_idx = np.argsort(robust)[:elite_count]
            for j in elite_idx:
                samples = [float(raw[j])]
                for _ in range(self.repeats - 1):
                    if used >= budget:
                        break
                    samples.append(float(objective(points[j])))
                    used += 1
                robust[j] = float(np.median(samples))

            # Re-clip after confirmation so extreme residuals cannot dominate fit.
            robust = _mad_clip(robust, self.clip_scale)
            j = int(np.argmin(robust))
            if robust[j] < best_f:
                best_x = points[j].copy()
                best_f = float(robust[j])

            if used + self.repeats <= budget:
                grad, hessian = _fit_relief(coords, robust)
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
                    samples = []
                    for _ in range(self.repeats):
                        if used >= budget:
                            break
                        samples.append(float(objective(cand)))
                        used += 1
                    if samples:
                        val = float(np.median(samples))
                        if val < best_f:
                            best_x, best_f = cand.copy(), val

        # Robust coordinate polish: each move is confirmed by a median of repeats.
        step = self.radius_schedule[-1] * span
        while used + 2 * dimension * self.repeats <= budget and step > 1e-13 * span:
            improved = False
            for axis in range(dimension):
                for sign in (-1.0, 1.0):
                    cand = best_x.copy()
                    cand[axis] = np.clip(cand[axis] + sign * step, lo, hi)
                    samples = [float(objective(cand)) for _ in range(self.repeats)]
                    used += self.repeats
                    val = float(np.median(samples))
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
            "RO-Hybrid-Multires-Robust-exp",
        )
