"""Experimental V5 with geometry-triggered local SpiralRelief refinement.

The global phase is V5 without coordinate collapse. A small pilot spiral probes
local curvature. Spiral refinement is activated only when the fitted Hessian is
positive definite and sufficiently anisotropic; otherwise a conservative
coordinate fallback uses the reserved evaluations.
"""
from __future__ import annotations

import numpy as np

from .common import Objective, OptimizationResult, validate_bounds
from .spiral_relief import SpiralReliefOptimizer, _fit_relief, _orthonormal_plane
from .v5 import ResolutiveV5


class ResolutiveV5ConditionalSpiral:
    def __init__(self, local_fraction: float = 0.15, anisotropy_threshold: float = 8.0) -> None:
        if not 0.08 <= local_fraction <= 0.35:
            raise ValueError("local_fraction must be in [0.08, 0.35]")
        if anisotropy_threshold <= 1.0:
            raise ValueError("anisotropy_threshold must be > 1")
        self.local_fraction = float(local_fraction)
        self.anisotropy_threshold = float(anisotropy_threshold)

    def minimize(self, objective: Objective, *, dimension: int,
                 bounds: tuple[float, float], budget: int = 6000,
                 seed: int = 0) -> OptimizationResult:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        lo, hi = validate_bounds(bounds)
        local_budget = max(160, int(round(budget * self.local_fraction)))
        global_budget = budget - local_budget
        base = ResolutiveV5(use_collapse=False).minimize(
            objective, dimension=dimension, bounds=(lo, hi),
            budget=global_budget, seed=seed,
        )
        used = base.evaluations
        remaining = budget - used
        if remaining < 30:
            return OptimizationResult(base.x, base.fun, used, seed, "RO-V5-ConditionalSpiral-exp")

        rng = np.random.default_rng(seed + 400009)
        span = hi - lo
        u, v = _orthonormal_plane(rng, dimension)
        probe_n = min(18, max(8, remaining // 4))
        radius = 0.025 * span
        idx = np.arange(1, probe_n + 1, dtype=float)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        radial = radius * np.sqrt(idx / probe_n)
        theta = golden_angle * idx * 2.5
        coords = np.column_stack((radial * np.cos(theta), radial * np.sin(theta)))
        points = np.clip(base.x + coords[:, :1] * u + coords[:, 1:] * v, lo, hi)
        vals = np.array([float(objective(p)) for p in points])
        used += probe_n
        best_i = int(np.argmin(vals))
        best_x = base.x.copy(); best_fun = float(base.fun)
        if vals[best_i] < best_fun:
            best_x = points[best_i].copy(); best_fun = float(vals[best_i])

        _, hessian = _fit_relief(coords, vals)
        eig = np.linalg.eigvalsh(hessian)
        positive = bool(np.all(eig > 1e-10))
        anis = float((eig[-1] + 1e-15) / (eig[0] + 1e-15)) if positive else 0.0
        remaining = budget - used

        if positive and anis >= self.anisotropy_threshold and remaining >= 20:
            local = SpiralReliefOptimizer(spiral_points=18, turns=2.5).refine(
                objective, start=best_x, bounds=(lo, hi), budget=remaining,
                seed=seed + 500009, radius_fraction=0.03,
            )
            used += local.evaluations
            if local.fun < best_fun:
                best_x, best_fun = local.x, float(local.fun)
        else:
            # Fallback: spend the remaining budget on a conservative coordinate
            # collapse, so the conditional method is not advantaged by using fewer
            # evaluations when the spiral gate stays closed.
            step = 0.03 * span
            while used + 2 * dimension <= budget and step > 1e-10 * span:
                improved = False
                for axis in range(dimension):
                    for sign in (-1.0, 1.0):
                        cand = best_x.copy()
                        cand[axis] = np.clip(cand[axis] + sign * step, lo, hi)
                        val = float(objective(cand)); used += 1
                        if val < best_fun:
                            best_x, best_fun = cand, val; improved = True
                if not improved:
                    step *= 0.5

        return OptimizationResult(best_x, best_fun, used, seed, "RO-V5-ConditionalSpiral-exp")
