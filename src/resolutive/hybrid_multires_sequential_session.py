"""Hybrid MultiResolution ask/tell session with sequentially faithful local dynamics."""
from __future__ import annotations

import numpy as np

from .hybrid_multires_session import HybridMultiResolutionSession
from .hybrid_sequential_session import HybridRegimeSequentialSession
from .session import AskBatch


class HybridMultiResolutionSequentialSession(HybridMultiResolutionSession):
    """Multires session preserving V2 and final coordinate-polish causality.

    The monolithic multires optimizer updates the incumbent immediately after
    each coordinate candidate.  Therefore the next candidate in the same sweep
    is generated from the updated incumbent.  This subclass preserves that
    dependency while keeping every objective evaluation external via ask/tell.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.core = HybridRegimeSequentialSession(
            dimension=self.dimension,
            bounds=(self.lo, self.hi),
            budget=self.core_budget,
            seed=self.seed,
            local_fraction=0.10,
        )
        self._polish_in_sweep = False
        self._polish_axis = 0
        self._polish_sign_index = 0
        self._polish_sweep_improved = False

    def ask(self) -> AskBatch:
        if self.phase != "polish":
            return super().ask()

        if self.pending is not None:
            raise RuntimeError("tell() must be called before the next ask()")
        if self.done:
            raise RuntimeError("optimization session is complete")
        if self.best_x is None:
            self.phase = "done"
            raise RuntimeError("optimization session is complete")

        # Match the monolithic while-condition at the start of every complete
        # 2*d coordinate sweep.  Once a sweep starts, finish it one candidate
        # at a time so accepted moves affect subsequent coordinates.
        if not self._polish_in_sweep:
            if (
                self.remaining < 2 * self.dimension
                or self.polish_step <= 1e-13 * self.span
            ):
                self.phase = "done"
                raise RuntimeError("optimization session is complete")
            self._polish_in_sweep = True
            self._polish_axis = 0
            self._polish_sign_index = 0
            self._polish_sweep_improved = False

        sign = -1.0 if self._polish_sign_index == 0 else 1.0
        candidate = self.best_x.copy()
        candidate[self._polish_axis] = np.clip(
            candidate[self._polish_axis] + sign * self.polish_step,
            self.lo,
            self.hi,
        )
        return self._emit(candidate.reshape(1, -1), "sequential_polish")

    def tell(self, values) -> None:
        if self.phase != "polish" or self.pending_kind != "sequential_polish":
            super().tell(values)
            return

        if self.pending is None:
            raise RuntimeError("ask() must be called before tell()")
        vals = np.asarray(values, dtype=float)
        if vals.shape != (1,):
            raise ValueError("values must contain one scalar for each asked point")
        if not np.all(np.isfinite(vals)):
            raise ValueError("values must be finite")

        point = self.pending[0].copy()
        value = float(vals[0])
        self.evaluations += 1
        self.pending = None
        self.pending_kind = None
        self.pending_coords = None

        if value < self.best_fun:
            self.best_fun = value
            self.best_x = point
            self._polish_sweep_improved = True

        if self._polish_sign_index == 0:
            self._polish_sign_index = 1
            return

        self._polish_sign_index = 0
        self._polish_axis += 1
        if self._polish_axis < self.dimension:
            return

        # Full sweep completed.  Shrink exactly when the monolithic policy
        # would shrink: only if no candidate in the complete sweep improved.
        if not self._polish_sweep_improved:
            self.polish_step *= 0.25
        self._polish_in_sweep = False
        self._polish_axis = 0
        self._polish_sign_index = 0

        if (
            self.remaining < 2 * self.dimension
            or self.polish_step <= 1e-13 * self.span
        ):
            self.phase = "done"
        else:
            self.phase = "polish"
