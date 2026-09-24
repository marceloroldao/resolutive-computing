"""Experimental early-refinement candidate for Resolutive Computing.

This module is intentionally outside the public API.  It tests the hypothesis
raised by the v1 efficiency campaign: the current Hybrid-Multires engine spends
too much of its budget in global/core exploration before exploiting a region
that is already promising.

The candidate preserves the existing ResolutiveHybridRegime core unchanged,
but hands control to an adaptive sequential coordinate refinement earlier.
The local step contracts only after a full unsuccessful sweep and may expand
slightly after a productive sweep.  Every accepted move is objectively
verified and immediately becomes the incumbent for subsequent coordinates.
"""
from __future__ import annotations

import numpy as np

from resolutive.optimization.common import Objective, OptimizationResult, validate_bounds
from resolutive.optimization.hybrid_regime import ResolutiveHybridRegime


class ResolutiveAdaptiveRefinementV1:
    """Experimental Hybrid -> adaptive coordinate-refinement optimizer.

    This is a research candidate, not part of the v1 compatibility surface.
    """

    def __init__(
        self,
        *,
        core_fraction: float = 0.60,
        initial_step_fraction: float = 0.08,
        contraction: float = 0.35,
        expansion: float = 1.20,
        max_step_fraction: float = 0.12,
        min_step_fraction: float = 1e-12,
    ) -> None:
        if not 0.55 <= core_fraction <= 0.80:
            raise ValueError("core_fraction must be in [0.55, 0.80]")
        if not 0.0 < initial_step_fraction <= max_step_fraction:
            raise ValueError("initial_step_fraction must be in (0, max_step_fraction]")
        if not 0.0 < contraction < 1.0:
            raise ValueError("contraction must be in (0, 1)")
        if expansion < 1.0:
            raise ValueError("expansion must be >= 1")
        if min_step_fraction <= 0.0:
            raise ValueError("min_step_fraction must be positive")
        self.core_fraction = float(core_fraction)
        self.initial_step_fraction = float(initial_step_fraction)
        self.contraction = float(contraction)
        self.expansion = float(expansion)
        self.max_step_fraction = float(max_step_fraction)
        self.min_step_fraction = float(min_step_fraction)

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
        if budget < 1000:
            raise ValueError("budget must be >= 1000")
        lo, hi = validate_bounds(bounds)
        span = hi - lo

        # RO-Hybrid requires at least 600 evaluations.  Keep the split explicit
        # and deterministic so candidate-vs-baseline comparisons are auditable.
        core_budget = max(600, int(round(budget * self.core_fraction)))
        core_budget = min(core_budget, budget - 2 * dimension)
        if core_budget < 600:
            raise ValueError("budget is too small for adaptive refinement")

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

        step = self.initial_step_fraction * span
        max_step = self.max_step_fraction * span
        min_step = self.min_step_fraction * span
        productive_sweeps = 0
        contractions = 0
        accepted_moves = 0

        while used + 2 * dimension <= budget and step > min_step:
            sweep_start = best_f
            sweep_moves = 0

            # Sequential causality is deliberate: an accepted move on axis i
            # changes the starting point for axis i+1 in the same sweep.
            for axis in range(dimension):
                for sign in (-1.0, 1.0):
                    if used >= budget:
                        break
                    cand = best_x.copy()
                    cand[axis] = np.clip(cand[axis] + sign * step, lo, hi)
                    val = float(objective(cand))
                    used += 1
                    if val < best_f:
                        best_x = cand
                        best_f = val
                        sweep_moves += 1
                        accepted_moves += 1
                if used >= budget:
                    break

            if best_f < sweep_start:
                productive_sweeps += 1
                # A productive scale remains relevant.  Expand only modestly so
                # the optimizer can traverse a basin without resetting globally.
                if sweep_moves >= max(1, dimension // 4):
                    step = min(max_step, step * self.expansion)
            else:
                step *= self.contraction
                contractions += 1

        return OptimizationResult(
            best_x,
            best_f,
            used,
            seed,
            "RO-Adaptive-Refinement-v1-exp",
            diagnostics={
                "core_budget": core_budget,
                "core_evaluations": int(core.evaluations),
                "accepted_moves": accepted_moves,
                "productive_sweeps": productive_sweeps,
                "contractions": contractions,
                "final_step_fraction": float(step / span),
                "protocol": "early-core-handoff-adaptive-coordinate",
            },
        )
