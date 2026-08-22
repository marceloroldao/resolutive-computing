"""Adaptive robust multiresolution optimizer.

The robust machinery is activated only when repeated evaluations at the current
best point show noise above a local scale threshold. Clean problems therefore
follow the ordinary multiresolution path as closely as possible, while noisy
problems gain median confirmation and MAD clipping.
"""
from __future__ import annotations

import numpy as np

from .common import Objective, OptimizationResult, validate_bounds
from .hybrid_regime import ResolutiveHybridRegime
from .hybrid_multires_robust import _mad_clip
from .spiral_relief import _fit_relief, _orthonormal_plane


class ResolutiveHybridMultiResolutionAdaptive:
    def __init__(
        self,
        local_fraction: float = 0.22,
        radius_schedule: tuple[float, ...] = (0.04, 0.01, 0.0025, 0.000625),
        points_per_level: int = 18,
        turns: float = 2.5,
        probe_repeats: int = 3,
        robust_repeats: int = 3,
        elite_resamples: int = 2,
        noise_ratio_threshold: float = 0.20,
        clip_scale: float = 3.0,
    ) -> None:
        if not 0.15 <= local_fraction <= 0.35:
            raise ValueError("local_fraction must be in [0.15, 0.35]")
        if len(radius_schedule) < 2 or any(r <= 0 for r in radius_schedule):
            raise ValueError("radius_schedule must contain at least two positive radii")
        if any(b >= a for a, b in zip(radius_schedule, radius_schedule[1:])):
            raise ValueError("radius_schedule must be strictly decreasing")
        if points_per_level < 8:
            raise ValueError("points_per_level must be >= 8")
        if probe_repeats < 3 or probe_repeats % 2 == 0:
            raise ValueError("probe_repeats must be odd and >= 3")
        if robust_repeats < 3 or robust_repeats % 2 == 0:
            raise ValueError("robust_repeats must be odd and >= 3")
        if elite_resamples < 1:
            raise ValueError("elite_resamples must be >= 1")
        if noise_ratio_threshold <= 0 or clip_scale <= 0:
            raise ValueError("thresholds must be positive")
        self.local_fraction = float(local_fraction)
        self.radius_schedule = tuple(float(r) for r in radius_schedule)
        self.points_per_level = int(points_per_level)
        self.turns = float(turns)
        self.probe_repeats = int(probe_repeats)
        self.robust_repeats = int(robust_repeats)
        self.elite_resamples = int(elite_resamples)
        self.noise_ratio_threshold = float(noise_ratio_threshold)
        self.clip_scale = float(clip_scale)

    @staticmethod
    def _noise_ratio(samples: list[float], local_scale: float) -> float:
        a = np.asarray(samples, dtype=float)
        spread = float(np.max(a) - np.min(a))
        return spread / max(float(local_scale), 1e-12)

    def minimize(self, objective: Objective, *, dimension: int,
                 bounds: tuple[float, float], budget: int = 6000,
                 seed: int = 0) -> OptimizationResult:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        lo, hi = validate_bounds(bounds)
        if budget < 1000:
            raise ValueError("budget must be >= 1000")

        local_budget = max(260, int(round(budget * self.local_fraction)))
        core_budget = budget - local_budget
        if core_budget < 600:
            core_budget = 600
        core = ResolutiveHybridRegime(local_fraction=0.10).minimize(
            objective, dimension=dimension, bounds=(lo, hi),
            budget=core_budget, seed=seed,
        )
        best_x = core.x.copy()
        best_f = float(core.fun)
        used = int(core.evaluations)
        span = hi - lo
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        for level, radius_fraction in enumerate(self.radius_schedule):
            remaining = budget - used
            levels_left = len(self.radius_schedule) - level
            if remaining < 12:
                break
            level_budget = max(10, remaining // levels_left)
            n = min(self.points_per_level, max(8, level_budget - self.probe_repeats))
            if used + n + self.probe_repeats > budget or n < 8:
                break

            rng = np.random.default_rng(seed + 1010001 + level * 9973)
            u, v = _orthonormal_plane(rng, dimension)
            idx = np.arange(1, n + 1, dtype=float)
            radial = radius_fraction * span * np.sqrt(idx / n)
            theta = golden_angle * idx * self.turns
            coords = np.column_stack((radial * np.cos(theta), radial * np.sin(theta)))
            points = np.clip(best_x + coords[:, :1] * u + coords[:, 1:] * v, lo, hi)
            raw = np.array([float(objective(p)) for p in points], dtype=float)
            used += n

            local_med = float(np.median(raw))
            local_mad = float(np.median(np.abs(raw - local_med)))
            local_scale = max(1.4826 * local_mad, 1e-12)

            probes = []
            for _ in range(self.probe_repeats):
                if used >= budget:
                    break
                probes.append(float(objective(best_x)))
                used += 1
            noisy = len(probes) >= 2 and self._noise_ratio(probes, local_scale) > self.noise_ratio_threshold

            vals = raw.copy()
            if noisy:
                vals = _mad_clip(vals, self.clip_scale)
                elite_count = min(self.elite_resamples, n)
                for j in np.argsort(vals)[:elite_count]:
                    samples = [float(raw[j])]
                    for _ in range(self.robust_repeats - 1):
                        if used >= budget:
                            break
                        samples.append(float(objective(points[j])))
                        used += 1
                    vals[j] = float(np.median(samples))
                vals = _mad_clip(vals, self.clip_scale)
                if probes:
                    best_f = min(best_f, float(np.median(probes)))

            j = int(np.argmin(vals))
            if vals[j] < best_f:
                best_x = points[j].copy()
                best_f = float(vals[j])

            confirm_cost = self.robust_repeats if noisy else 1
            if used + confirm_cost <= budget:
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
                        step2 *= max_step / (norm + 1e-15)
                    cand = np.clip(best_x + step2[0] * u + step2[1] * v, lo, hi)
                    samples = []
                    for _ in range(confirm_cost):
                        if used >= budget:
                            break
                        samples.append(float(objective(cand)))
                        used += 1
                    if samples:
                        val = float(np.median(samples))
                        if val < best_f:
                            best_x, best_f = cand.copy(), val

        # Keep final polish cheap; noise gating has already protected local geometry.
        if used + 2 * dimension <= budget:
            step = self.radius_schedule[-1] * span
            while used + 2 * dimension <= budget and step > 1e-13 * span:
                improved = False
                for axis in range(dimension):
                    for sign in (-1.0, 1.0):
                        cand = best_x.copy()
                        cand[axis] = np.clip(cand[axis] + sign * step, lo, hi)
                        val = float(objective(cand)); used += 1
                        if val < best_f:
                            best_x, best_f = cand, val; improved = True
                if not improved:
                    step *= 0.25

        return OptimizationResult(best_x, best_f, used, seed,
                                  "RO-Hybrid-Multires-Adaptive-exp")
