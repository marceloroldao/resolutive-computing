"""Experimental ask/tell session for external black-box evaluation.

This establishes the transport/API contract needed for hardware, simulators and
remote services.  The current proposal policy is deliberately simple and is not
yet claimed to be equivalent to any Resolutive specialist engine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .optimization.common import OptimizationResult, validate_bounds


@dataclass(frozen=True)
class AskBatch:
    points: np.ndarray
    generation: int


class OptimizationSession:
    """Stateful experimental ask/tell optimizer for externally evaluated points."""

    def __init__(
        self,
        *,
        dimension: int,
        bounds: tuple[float, float],
        budget: int,
        seed: int = 0,
        batch_size: int = 8,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        if budget < 1:
            raise ValueError("budget must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.dimension = int(dimension)
        self.lo, self.hi = validate_bounds(bounds)
        self.budget = int(budget)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self._rng = np.random.default_rng(seed)
        self._generation = 0
        self._evaluations = 0
        self._pending: np.ndarray | None = None
        self._best_x: np.ndarray | None = None
        self._best_fun = float("inf")

    @property
    def evaluations(self) -> int:
        return self._evaluations

    @property
    def remaining(self) -> int:
        return self.budget - self._evaluations

    @property
    def done(self) -> bool:
        return self.remaining <= 0

    def ask(self, n: int | None = None) -> AskBatch:
        if self._pending is not None:
            raise RuntimeError("tell() must be called before the next ask()")
        if self.done:
            raise RuntimeError("optimization budget exhausted")
        count = min(self.remaining, self.batch_size if n is None else int(n))
        if count < 1:
            raise ValueError("ask size must be >= 1")

        if self._best_x is None:
            points = self._rng.uniform(self.lo, self.hi, size=(count, self.dimension))
        else:
            # Coarse-to-fine proposal scale: shrink with completed generations,
            # while retaining one global exploratory point when possible.
            span = self.hi - self.lo
            scale = max(0.02, 0.35 * (0.72 ** self._generation)) * span
            points = self._best_x + self._rng.normal(0.0, scale, size=(count, self.dimension))
            points = np.clip(points, self.lo, self.hi)
            if count > 1:
                points[0] = self._rng.uniform(self.lo, self.hi, size=self.dimension)

        self._pending = np.asarray(points, dtype=float)
        return AskBatch(self._pending.copy(), self._generation)

    def tell(self, values: np.ndarray | list[float]) -> None:
        if self._pending is None:
            raise RuntimeError("ask() must be called before tell()")
        vals = np.asarray(values, dtype=float)
        if vals.shape != (len(self._pending),):
            raise ValueError("values must contain one scalar for each asked point")
        if not np.all(np.isfinite(vals)):
            raise ValueError("values must be finite")

        idx = int(np.argmin(vals))
        if float(vals[idx]) < self._best_fun:
            self._best_fun = float(vals[idx])
            self._best_x = self._pending[idx].copy()

        self._evaluations += len(self._pending)
        self._pending = None
        self._generation += 1

    def result(self) -> OptimizationResult:
        if self._best_x is None:
            raise RuntimeError("no observations have been supplied")
        status = "success" if self.done else "running"
        return OptimizationResult(
            self._best_x.copy(),
            self._best_fun,
            self._evaluations,
            self.seed,
            "RO-AskTell-Prototype",
            status=status,
            diagnostics={
                "generation": self._generation,
                "remaining_budget": self.remaining,
                "protocol": "ask-tell-experimental",
            },
        )
