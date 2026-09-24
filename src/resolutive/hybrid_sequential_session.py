"""Causally faithful ask/tell port of the regime-selecting Resolutive hybrid.

This variant keeps the reconnaissance and geometry gate of HybridRegimeSession,
but replaces both sources of sequential dependence from the monolithic engine:
(1) the V2 specialist uses V2SequentialSession, and (2) the fallback coordinate
collapse emits one objective evaluation at a time and updates the incumbent
before the next coordinate/sign probe is constructed.
"""
from __future__ import annotations

import numpy as np

from .hybrid_session import HybridRegimeSession
from .session import AskBatch
from .v2_sequential_session import V2SequentialSession


class HybridRegimeSequentialSession(HybridRegimeSession):
    """Hybrid ask/tell session preserving monolithic sequential causality."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._seq_axis = 0
        self._seq_sign_index = 0
        self._seq_sweep_improved = False
        self._seq_signs = (-1.0, 1.0)

    def _reset_collapse_sweep(self) -> None:
        self._seq_axis = 0
        self._seq_sign_index = 0
        self._seq_sweep_improved = False

    def ask(self) -> AskBatch:
        if self.phase != "collapse":
            return super().ask()
        if self.pending is not None:
            raise RuntimeError("tell() must be called before the next ask()")
        if self.best_x is None:
            self.phase = "done"
            raise RuntimeError("optimization session is complete")

        # Match the monolithic outer while condition: begin a new coordinate
        # sweep only when all 2*d evaluations fit in the remaining budget.
        if self._seq_axis == 0 and self._seq_sign_index == 0:
            if self.remaining < 2 * self.dimension or self.collapse_step <= 1e-11 * self.span:
                self.phase = "done"
                raise RuntimeError("optimization session is complete")

        sign = self._seq_signs[self._seq_sign_index]
        candidate = self.best_x.copy()
        candidate[self._seq_axis] = np.clip(
            candidate[self._seq_axis] + sign * self.collapse_step,
            self.lo,
            self.hi,
        )
        return self._emit(candidate[None, :], "collapse-sequential")

    def tell(self, values) -> None:
        was_recon = self.phase == "recon"

        if self.phase == "collapse" and self.pending_kind == "collapse-sequential":
            if self.pending is None:
                raise RuntimeError("ask() must be called before tell()")
            vals = np.asarray(values, dtype=float)
            if vals.shape != (1,) or not np.all(np.isfinite(vals)):
                raise ValueError("values must contain one finite scalar for each asked point")

            point = self.pending[0].copy()
            value = float(vals[0])
            self.evaluations += 1
            self.pending = None
            self.pending_kind = None

            if value < self.best_fun:
                self.best_fun = value
                self.best_x = point.copy()
                self._seq_sweep_improved = True

            self._seq_sign_index += 1
            if self._seq_sign_index >= 2:
                self._seq_sign_index = 0
                self._seq_axis += 1

            if self._seq_axis >= self.dimension:
                if not self._seq_sweep_improved:
                    self.collapse_step *= 0.5
                self._reset_collapse_sweep()
                if self.remaining < 2 * self.dimension or self.collapse_step <= 1e-11 * self.span:
                    self.phase = "done"
            return

        super().tell(values)

        if was_recon and self.core_name == "V2":
            self.core = V2SequentialSession(
                dimension=self.dimension,
                bounds=(self.lo, self.hi),
                budget=self.core_budget,
                seed=self.seed,
            )

        # The gate may have just selected the fallback collapse path. Reset the
        # per-sweep causal state so the first sequential ask starts at axis 0.
        if self.phase == "collapse" and self._seq_axis == 0 and self._seq_sign_index == 0:
            self._seq_sweep_improved = False
