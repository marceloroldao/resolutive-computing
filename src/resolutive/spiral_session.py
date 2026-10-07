"""Stateful ask/tell port of local SpiralRelief refinement."""
from __future__ import annotations

import numpy as np

from .optimization.common import OptimizationResult, validate_bounds
from .optimization.spiral_relief import _fit_relief, _orthonormal_plane, _slide_vector
from .session import AskBatch


class SpiralRefineSession:
    """Incremental equivalent of ``SpiralReliefOptimizer.refine``."""

    def __init__(self, *, start: np.ndarray, bounds: tuple[float, float], budget: int,
                 seed: int = 0, spiral_points: int = 18, turns: float = 2.5,
                 radius_fraction: float = 0.04) -> None:
        self.lo, self.hi = validate_bounds(bounds)
        self.start = np.asarray(start, dtype=float).copy()
        self.dimension = len(self.start)
        if self.dimension < 2:
            raise ValueError("dimension must be >= 2")
        if budget <= spiral_points + 1:
            raise ValueError("budget too small for one spiral cycle")
        self.budget = int(budget)
        self.seed = int(seed)
        self.spiral_points = int(spiral_points)
        self.turns = float(turns)
        self.span = self.hi - self.lo
        self.radius = float(radius_fraction) * self.span
        self.min_radius = 1e-7 * self.span
        self.rng = np.random.default_rng(seed)

        self.center = np.clip(self.start, self.lo, self.hi)
        self.center_value: float | None = None
        self.best = self.center.copy()
        self.best_value = float("inf")
        self.used = 0
        self.stall = 0
        self.generation = 0
        self.phase = "center"
        self.pending: np.ndarray | None = None
        self.pending_kind: str | None = None
        self.coords: np.ndarray | None = None
        self.plane: tuple[np.ndarray, np.ndarray] | None = None
        self.mapped_point: np.ndarray | None = None
        self.mapped_value: float | None = None

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    @property
    def done(self) -> bool:
        return self.phase == "done" or self.remaining <= 0

    def _emit(self, points: np.ndarray, kind: str) -> AskBatch:
        self.pending = np.asarray(points, dtype=float)
        self.pending_kind = kind
        return AskBatch(self.pending.copy(), self.generation)

    def ask(self) -> AskBatch:
        if self.pending is not None:
            raise RuntimeError("tell() must be called before the next ask()")
        if self.done:
            raise RuntimeError("optimization session is complete")
        if self.phase == "center":
            return self._emit(self.center.reshape(1, -1), "center")
        if self.phase == "spiral":
            if self.used + self.spiral_points + 1 > self.budget or self.radius < self.min_radius:
                self.phase = "done"
                raise RuntimeError("optimization session is complete")
            u, v = _orthonormal_plane(self.rng, self.dimension)
            self.plane = (u, v)
            idx = np.arange(1, self.spiral_points + 1, dtype=float)
            radial = self.radius * np.sqrt(idx / self.spiral_points)
            theta = np.pi * (3.0 - np.sqrt(5.0)) * idx * self.turns
            self.coords = np.column_stack((radial * np.cos(theta), radial * np.sin(theta)))
            points = np.clip(self.center + self.coords[:, :1] * u + self.coords[:, 1:] * v,
                             self.lo, self.hi)
            return self._emit(points, "spiral")
        if self.phase == "slide":
            assert self.coords is not None and self.plane is not None and self._last_vals is not None
            grad, hessian = _fit_relief(self.coords, self._last_vals)
            slide2 = _slide_vector(grad, hessian, self.radius)
            u, v = self.plane
            candidate = np.clip(self.center + slide2[0] * u + slide2[1] * v, self.lo, self.hi)
            return self._emit(candidate.reshape(1, -1), "slide")
        raise RuntimeError(f"unknown phase: {self.phase}")

    def tell(self, values) -> None:
        if self.pending is None or self.pending_kind is None:
            raise RuntimeError("ask() must be called before tell()")
        vals = np.asarray(values, dtype=float)
        if vals.shape != (len(self.pending),) or not np.all(np.isfinite(vals)):
            raise ValueError("values must contain one finite scalar for each asked point")
        kind, points = self.pending_kind, self.pending.copy()
        self.used += len(points)
        self.pending = None
        self.pending_kind = None

        if kind == "center":
            self.center_value = float(vals[0])
            self.best_value = self.center_value
            self.best = self.center.copy()
            self.phase = "spiral"
            return
        if kind == "spiral":
            self._last_vals = vals.copy()
            j = int(np.argmin(vals))
            self.mapped_value = float(vals[j])
            self.mapped_point = points[j].copy()
            if self.mapped_value < self.best_value:
                self.best_value = self.mapped_value
                self.best = self.mapped_point.copy()
            self.phase = "slide"
            return
        if kind == "slide":
            assert self.center_value is not None and self.mapped_value is not None and self.mapped_point is not None
            candidate_value = float(vals[0])
            candidate = points[0].copy()
            if candidate_value < self.best_value:
                self.best_value = candidate_value
                self.best = candidate.copy()
            if min(self.mapped_value, candidate_value) < self.center_value:
                if candidate_value <= self.mapped_value:
                    self.center, self.center_value = candidate, candidate_value
                else:
                    self.center, self.center_value = self.mapped_point.copy(), self.mapped_value
                self.radius *= 0.92
                self.stall = 0
            else:
                self.stall += 1
                self.radius *= 0.72 if self.stall >= 2 else 0.90
                if self.stall >= 2:
                    self.stall = 0
            self.generation += 1
            self.phase = "spiral"
            if self.used + self.spiral_points + 1 > self.budget or self.radius < self.min_radius:
                self.phase = "done"

    def result(self) -> OptimizationResult:
        if self.center_value is None:
            raise RuntimeError("no observations have been supplied")
        return OptimizationResult(
            self.best.copy(), self.best_value, self.used, self.seed,
            "SpiralRelief-local-AskTell",
            status="success" if self.done else "running",
            diagnostics={"phase": self.phase, "generation": self.generation,
                         "remaining_budget": self.remaining, "protocol": "ask-tell-spiral"},
        )
