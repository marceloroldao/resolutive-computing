"""Regime-gated adaptive refinement experiment.

V2 keeps the public engine untouched. It uses the existing HybridRegime core,
then performs two deterministic coordinate probes at different scales. Only if
both scales demonstrate productive local structure does it commit the remaining
budget to sequential adaptive refinement. Otherwise it falls back to a compact
multiresolution spiral/local-polish phase.

The gate is benchmark-agnostic: it depends only on observed objective response.
"""
from __future__ import annotations

import numpy as np

from resolutive.optimization.common import Objective, OptimizationResult, validate_bounds
from resolutive.optimization.hybrid_regime import ResolutiveHybridRegime
from resolutive.optimization.spiral_relief import _fit_relief, _orthonormal_plane


class ResolutiveAdaptiveRefinementV2:
    def __init__(
        self,
        *,
        core_fraction: float = 0.75,
        coarse_step_fraction: float = 0.04,
        fine_step_fraction: float = 0.01,
        contraction: float = 0.35,
        expansion: float = 1.15,
    ) -> None:
        self.core_fraction = float(core_fraction)
        self.coarse_step_fraction = float(coarse_step_fraction)
        self.fine_step_fraction = float(fine_step_fraction)
        self.contraction = float(contraction)
        self.expansion = float(expansion)

    @staticmethod
    def _coordinate_sweep(objective, x, best_f, step, lo, hi, budget_left):
        best_x = x.copy()
        used = 0
        moves = 0
        start_f = float(best_f)
        dimension = len(best_x)
        for axis in range(dimension):
            for sign in (-1.0, 1.0):
                if used >= budget_left:
                    return best_x, best_f, used, moves, start_f
                cand = best_x.copy()
                cand[axis] = np.clip(cand[axis] + sign * step, lo, hi)
                val = float(objective(cand))
                used += 1
                if val < best_f:
                    best_x, best_f = cand, val
                    moves += 1
        return best_x, best_f, used, moves, start_f

    def minimize(self, objective: Objective, *, dimension: int, bounds: tuple[float, float], budget: int = 6000, seed: int = 0) -> OptimizationResult:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        if budget < 1000:
            raise ValueError("budget must be >= 1000")
        lo, hi = validate_bounds(bounds)
        span = hi - lo

        reserve = max(4 * dimension, 120)
        core_budget = max(600, int(round(budget * self.core_fraction)))
        core_budget = min(core_budget, budget - reserve)
        core = ResolutiveHybridRegime(local_fraction=0.10).minimize(
            objective, dimension=dimension, bounds=(lo, hi), budget=core_budget, seed=seed
        )
        best_x = core.x.copy()
        best_f = float(core.fun)
        used = int(core.evaluations)

        # Two-scale observation gate. A locally resolvable basin should remain
        # productive when the scale contracts; a spurious coarse improvement
        # should not be enough to switch the whole remaining budget.
        coarse_start = best_f
        best_x, best_f, u, coarse_moves, _ = self._coordinate_sweep(
            objective, best_x, best_f, self.coarse_step_fraction * span, lo, hi, budget - used
        )
        used += u
        coarse_gain = max(0.0, coarse_start - best_f) / max(abs(coarse_start), 1.0)

        fine_start = best_f
        best_x, best_f, u, fine_moves, _ = self._coordinate_sweep(
            objective, best_x, best_f, self.fine_step_fraction * span, lo, hi, budget - used
        )
        used += u
        fine_gain = max(0.0, fine_start - best_f) / max(abs(fine_start), 1.0)

        min_moves = max(1, dimension // 10)
        local_regime = (
            coarse_moves >= min_moves
            and fine_moves >= min_moves
            and (coarse_gain + fine_gain) > 1e-5
        )

        if local_regime:
            step = self.fine_step_fraction * span
            max_step = self.coarse_step_fraction * span
            min_step = 1e-12 * span
            while used + 2 * dimension <= budget and step > min_step:
                sweep_start = best_f
                best_x, best_f, u, moves, _ = self._coordinate_sweep(
                    objective, best_x, best_f, step, lo, hi, budget - used
                )
                used += u
                if best_f < sweep_start:
                    if moves >= max(1, dimension // 6):
                        step = min(max_step, step * self.expansion)
                else:
                    step *= self.contraction
        else:
            # Conservative fallback: preserve a multiresolution local map rather
            # than forcing coordinate descent when the two-scale gate is weak.
            golden_angle = np.pi * (3.0 - np.sqrt(5.0))
            radii = (0.04, 0.01, 0.0025, 0.000625)
            for level, radius_fraction in enumerate(radii):
                remaining = budget - used
                if remaining < 10:
                    break
                n = min(18, remaining - 1)
                if n < 8:
                    break
                rng = np.random.default_rng(seed + 910001 + level * 9973)
                uvec, vvec = _orthonormal_plane(rng, dimension)
                idx = np.arange(1, n + 1, dtype=float)
                radial = radius_fraction * span * np.sqrt(idx / n)
                theta = golden_angle * idx * 2.5
                coords = np.column_stack((radial * np.cos(theta), radial * np.sin(theta)))
                points = np.clip(best_x + coords[:, :1] * uvec + coords[:, 1:] * vvec, lo, hi)
                vals = np.asarray([float(objective(p)) for p in points], dtype=float)
                used += n
                j = int(np.argmin(vals))
                if vals[j] < best_f:
                    best_x, best_f = points[j].copy(), float(vals[j])
                if used < budget:
                    grad, hessian = _fit_relief(coords, vals)
                    step2 = None
                    if np.all(np.linalg.eigvalsh(hessian) > 1e-10):
                        try:
                            step2 = -np.linalg.solve(hessian + 1e-10 * np.eye(2), grad)
                        except np.linalg.LinAlgError:
                            step2 = None
                    if step2 is not None and np.all(np.isfinite(step2)):
                        radius = radius_fraction * span
                        norm = float(np.linalg.norm(step2))
                        if norm > radius:
                            step2 *= radius / (norm + 1e-15)
                        cand = np.clip(best_x + step2[0] * uvec + step2[1] * vvec, lo, hi)
                        val = float(objective(cand)); used += 1
                        if val < best_f:
                            best_x, best_f = cand, val

        return OptimizationResult(
            best_x, best_f, used, seed, "RO-Adaptive-Refinement-v2-exp",
            diagnostics={
                "core_budget": core_budget,
                "local_regime": bool(local_regime),
                "coarse_moves": int(coarse_moves),
                "fine_moves": int(fine_moves),
                "coarse_gain": float(coarse_gain),
                "fine_gain": float(fine_gain),
                "protocol": "two-scale-regime-gated-refinement",
            },
        )
