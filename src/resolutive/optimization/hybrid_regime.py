"""Experimental regime-selecting Resolutive hybrid.

A small black-box reconnaissance stage estimates local ruggedness from paired
radial probes. The controller then selects either RO-V2 (more conservative and
robust under noisy/rugged landscapes) or RO-V5 (strong multimodal reference),
followed by a geometry-gated local refinement. The selector uses only observed
objective values; it does not know the benchmark identity.
"""
from __future__ import annotations

import numpy as np

from .common import Objective, OptimizationResult, validate_bounds
from .spiral_relief import SpiralReliefOptimizer, _fit_relief, _orthonormal_plane
from .v2 import ResolutiveV2
from .v5 import ResolutiveV5


def _reconnaissance(objective: Objective, *, dimension: int,
                    bounds: tuple[float, float], budget: int,
                    seed: int) -> tuple[float, np.ndarray, float, int]:
    """Estimate landscape ruggedness with paired radial samples.

    Returns (ruggedness, best_x, best_f, evaluations). Ruggedness is the
    fraction of sampled directions whose radial finite differences change sign,
    which is a scale-free indication of local non-monotonicity.
    """
    lo, hi = validate_bounds(bounds)
    rng = np.random.default_rng(seed)
    span = hi - lo
    center = rng.uniform(lo, hi, dimension)
    center_f = float(objective(center))
    best_x, best_f = center.copy(), center_f
    used = 1
    directions = max(4, min(16, (budget - 1) // 3))
    sign_changes = 0
    valid = 0
    for _ in range(directions):
        if used + 3 > budget:
            break
        d = rng.normal(size=dimension)
        d /= np.linalg.norm(d) + 1e-15
        fs = []
        for frac in (0.04, 0.10, 0.20):
            x = np.clip(center + frac * span * d, lo, hi)
            fx = float(objective(x)); used += 1
            fs.append(fx)
            if fx < best_f:
                best_x, best_f = x.copy(), fx
        delta1, delta2 = fs[1] - fs[0], fs[2] - fs[1]
        if np.isfinite(delta1) and np.isfinite(delta2):
            valid += 1
            if delta1 * delta2 < 0.0:
                sign_changes += 1
    ruggedness = sign_changes / max(1, valid)
    return float(ruggedness), best_x, float(best_f), used


class ResolutiveHybridRegime:
    """Black-box regime selector with conditional geometry-aware refinement."""

    def __init__(self, reconnaissance_fraction: float = 0.04,
                 local_fraction: float = 0.14,
                 ruggedness_threshold: float = 0.28,
                 anisotropy_threshold: float = 8.0) -> None:
        if not 0.02 <= reconnaissance_fraction <= 0.10:
            raise ValueError("reconnaissance_fraction must be in [0.02, 0.10]")
        if not 0.08 <= local_fraction <= 0.25:
            raise ValueError("local_fraction must be in [0.08, 0.25]")
        self.reconnaissance_fraction = float(reconnaissance_fraction)
        self.local_fraction = float(local_fraction)
        self.ruggedness_threshold = float(ruggedness_threshold)
        self.anisotropy_threshold = float(anisotropy_threshold)

    def minimize(self, objective: Objective, *, dimension: int,
                 bounds: tuple[float, float], budget: int = 6000,
                 seed: int = 0) -> OptimizationResult:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        lo, hi = validate_bounds(bounds)
        if budget < 600:
            raise ValueError("budget must be >= 600")

        recon_budget = max(40, int(round(budget * self.reconnaissance_fraction)))
        local_budget = max(140, int(round(budget * self.local_fraction)))
        core_budget = budget - recon_budget - local_budget
        ruggedness, probe_x, probe_f, recon_used = _reconnaissance(
            objective, dimension=dimension, bounds=(lo, hi),
            budget=recon_budget, seed=seed + 700001,
        )

        # Rugged landscapes favor V2's steadier exploration; smoother landscapes
        # use V5, which has been the stronger general clean-function reference.
        if ruggedness >= self.ruggedness_threshold:
            core = ResolutiveV2().minimize(
                objective, dimension=dimension, bounds=(lo, hi),
                budget=core_budget, seed=seed,
            )
            core_name = "V2"
        else:
            core = ResolutiveV5(use_collapse=False).minimize(
                objective, dimension=dimension, bounds=(lo, hi),
                budget=core_budget, seed=seed,
            )
            core_name = "V5"

        best_x, best_f = (core.x.copy(), float(core.fun))
        if probe_f < best_f:
            best_x, best_f = probe_x.copy(), probe_f
        used = recon_used + core.evaluations
        remaining = budget - used
        if remaining < 30:
            return OptimizationResult(best_x, best_f, used, seed, f"RO-Hybrid-exp[{core_name}]")

        # Geometry gate from a small spiral pilot around the incumbent.
        rng = np.random.default_rng(seed + 700019)
        span = hi - lo
        u, v = _orthonormal_plane(rng, dimension)
        probe_n = min(18, max(8, remaining // 5))
        radius = 0.025 * span
        idx = np.arange(1, probe_n + 1, dtype=float)
        theta = np.pi * (3.0 - np.sqrt(5.0)) * idx * 2.5
        radial = radius * np.sqrt(idx / probe_n)
        coords = np.column_stack((radial * np.cos(theta), radial * np.sin(theta)))
        points = np.clip(best_x + coords[:, :1] * u + coords[:, 1:] * v, lo, hi)
        vals = np.array([float(objective(p)) for p in points]); used += probe_n
        j = int(np.argmin(vals))
        if vals[j] < best_f:
            best_x, best_f = points[j].copy(), float(vals[j])
        _, hessian = _fit_relief(coords, vals)
        eig = np.linalg.eigvalsh(hessian)
        positive = bool(np.all(eig > 1e-10))
        anis = float((eig[-1] + 1e-15) / (eig[0] + 1e-15)) if positive else 0.0
        remaining = budget - used

        if positive and anis >= self.anisotropy_threshold and remaining >= 20:
            local = SpiralReliefOptimizer(spiral_points=18, turns=2.5).refine(
                objective, start=best_x, bounds=(lo, hi), budget=remaining,
                seed=seed + 700027, radius_fraction=0.03,
            )
            used += local.evaluations
            if local.fun < best_f:
                best_x, best_f = local.x.copy(), float(local.fun)
        else:
            # Precision fallback: axis-local collapse with shrinking radius.
            step = 0.025 * span
            while used + 2 * dimension <= budget and step > 1e-11 * span:
                improved = False
                for axis in range(dimension):
                    for sign in (-1.0, 1.0):
                        cand = best_x.copy()
                        cand[axis] = np.clip(cand[axis] + sign * step, lo, hi)
                        val = float(objective(cand)); used += 1
                        if val < best_f:
                            best_x, best_f = cand, val; improved = True
                if not improved:
                    step *= 0.5

        return OptimizationResult(
            best_x, best_f, used, seed,
            f"RO-Hybrid-exp[{core_name};rug={ruggedness:.3f}]",
        )
