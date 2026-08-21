"""Causally faithful ask/tell refinement for Resolutive V2.

The monolithic V2 coordinate-collapse stage evaluates one coordinate move at a
time and immediately updates the incumbent before evaluating the next move.
This subclass preserves that sequential dependency while keeping every objective
evaluation external through ask()/tell().
"""
from __future__ import annotations

import numpy as np

from .session import AskBatch
from .specialist_sessions import V2Session


class V2SequentialSession(V2Session):
    """V2 ask/tell session with monolithic-equivalent sequential collapse."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._seq_axis = 0
        self._seq_sign_index = 0
        self._seq_sweep_improved = False
        self._seq_signs = (-1.0, 1.0)

    def _reset_sweep(self) -> None:
        self._seq_axis = 0
        self._seq_sign_index = 0
        self._seq_sweep_improved = False

    def ask(self) -> AskBatch:
        if self._phase != "collapse":
            return super().ask()
        if self._pending is not None:
            raise RuntimeError("tell() must be called before the next ask()")
        if self._collapse_point is None:
            if self.best_x is None:
                raise RuntimeError("no incumbent available for collapse")
            self._collapse_point = self.best_x.copy()

        # The monolithic loop starts a sweep only when the entire 2*d sweep
        # fits in the remaining budget.
        if self._seq_axis == 0 and self._seq_sign_index == 0:
            if self.remaining < 2 * self.dimension or self._collapse_step <= 1e-10 * self.span:
                self._phase = "done"
                raise RuntimeError("optimization session is complete")

        sign = self._seq_signs[self._seq_sign_index]
        candidate = self._collapse_point.copy()
        candidate[self._seq_axis] = np.clip(
            candidate[self._seq_axis] + sign * self._collapse_step,
            self.lo,
            self.hi,
        )
        return self._set_pending(candidate[None, :], "collapse-sequential")

    def tell(self, values) -> None:
        if self._phase != "collapse" or self._pending_kind != "collapse-sequential":
            return super().tell(values)
        if self._pending is None:
            raise RuntimeError("ask() must be called before tell()")

        vals = np.asarray(values, dtype=float)
        if vals.shape != (1,) or not np.all(np.isfinite(vals)):
            raise ValueError("values must contain one finite scalar for each asked point")

        point = self._pending[0].copy()
        value = float(vals[0])
        self._evaluations += 1
        self._pending = None
        self._pending_kind = None

        if value < self.best_fun:
            self.best_fun = value
            self.best_x = point.copy()
            self._collapse_point = point.copy()
            self._seq_sweep_improved = True

        self._seq_sign_index += 1
        if self._seq_sign_index >= 2:
            self._seq_sign_index = 0
            self._seq_axis += 1

        if self._seq_axis >= self.dimension:
            if not self._seq_sweep_improved:
                self._collapse_step *= 0.5
            self._reset_sweep()
            if self.remaining < 2 * self.dimension or self._collapse_step <= 1e-10 * self.span:
                self._phase = "done"
