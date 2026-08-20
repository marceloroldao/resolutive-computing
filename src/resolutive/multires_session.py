"""Stateful multiresolution ask/tell optimizer.

This is the first incremental Resolutive engine intended for external evaluation.
It preserves optimization state across ask/tell boundaries and mirrors the
coarse-to-fine local geometry used by the monolithic Hybrid-Multires engine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .optimization.common import OptimizationResult, validate_bounds
from .optimization.spiral_relief import _fit_relief, _orthonormal_plane
from .session import AskBatch


@dataclass(frozen=True)
class MultiResolutionState:
    phase: str
    level: int
    generation: int
    evaluations: int


class MultiResolutionSession:
    """Incremental coarse-to-fine optimizer for external objective evaluation."""

    def __init__(
        self,
        *,
        dimension: int,
        bounds: tuple[float, float],
        budget: int,
        seed: int = 0,
        batch_size: int = 16,
        exploration_fraction: float = 0.55,
        radius_schedule: tuple[float, ...] = (0.04, 0.01, 0.0025, 0.000625),
        points_per_level: int = 18,
        turns: float = 2.5,
    ) -> None:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        if budget < 200:
            raise ValueError("budget must be >= 200")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.25 <= exploration_fraction <= 0.80:
            raise ValueError("exploration_fraction must be in [0.25, 0.80]")
        if len(radius_schedule) < 2 or any(r <= 0 for r in radius_schedule):
            raise ValueError("radius_schedule must contain at least two positive radii")
        if any(b >= a for a, b in zip(radius_schedule, radius_schedule[1:])):
            raise ValueError("radius_schedule must be strictly decreasing")
        if points_per_level < 8:
            raise ValueError("points_per_level must be >= 8")

        self.dimension = int(dimension)
        self.lo, self.hi = validate_bounds(bounds)
        self.budget = int(budget)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.exploration_budget = max(batch_size, int(round(budget * exploration_fraction)))
        self.radius_schedule = tuple(float(r) for r in radius_schedule)
        self.points_per_level = int(points_per_level)
        self.turns = float(turns)

        self._rng = np.random.default_rng(seed + 712_001)
        self._evaluations = 0
        self._generation = 0
        self._level = 0
        self._phase = "explore"
        self._pending: np.ndarray | None = None
        self._pending_kind: str | None = None
        self._pending_coords: np.ndarray | None = None
        self._plane: tuple[np.ndarray, np.ndarray] | None = None
        self._best_x: np.ndarray | None = None
        self._best_fun = float("inf")
        self._last_spiral_values: np.ndarray | None = None

    @property
    def evaluations(self) -> int:
        return self._evaluations

    @property
    def remaining(self) -> int:
        return self.budget - self._evaluations

    @property
    def done(self) -> bool:
        return self.remaining <= 0 or self._phase == "done"

    @property
    def state(self) -> MultiResolutionState:
        return MultiResolutionState(self._phase, self._level, self._generation, self._evaluations)

    def _set_pending(self, points: np.ndarray, kind: str, coords: np.ndarray | None = None) -> AskBatch:
        self._pending = np.asarray(points, dtype=float)
        self._pending_kind = kind
        self._pending_coords = None if coords is None else np.asarray(coords, dtype=float)
        return AskBatch(self._pending.copy(), self._generation)

    def ask(self) -> AskBatch:
        if self._pending is not None:
            raise RuntimeError("tell() must be called before the next ask()")
        if self.done:
            raise RuntimeError("optimization session is complete")

        if self._phase == "explore":
            remaining_explore = max(0, self.exploration_budget - self._evaluations)
            if remaining_explore == 0:
                self._phase = "spiral"
                return self.ask()
            n = min(self.batch_size, self.remaining, remaining_explore)
            points = self._rng.uniform(self.lo, self.hi, size=(n, self.dimension))
            return self._set_pending(points, "explore")

        if self._phase == "spiral":
            if self._best_x is None:
                self._phase = "done"
                return self.ask()
            if self._level >= len(self.radius_schedule) or self.remaining < 2:
                self._phase = "polish"
                return self.ask()
            n = min(self.points_per_level, self.remaining - 1 if self.remaining > 1 else 1)
            if n < 1:
                self._phase = "done"
                return self.ask()
            rng = np.random.default_rng(self.seed + 810_001 + self._level * 9_973)
            u, v = _orthonormal_plane(rng, self.dimension)
            self._plane = (u, v)
            idx = np.arange(1, n + 1, dtype=float)
            span = self.hi - self.lo
            radius = self.radius_schedule[self._level]
            radial = radius * span * np.sqrt(idx / n)
            theta = np.pi * (3.0 - np.sqrt(5.0)) * idx * self.turns
            coords = np.column_stack((radial * np.cos(theta), radial * np.sin(theta)))
            points = np.clip(self._best_x + coords[:, :1] * u + coords[:, 1:] * v, self.lo, self.hi)
            return self._set_pending(points, "spiral", coords)

        if self._phase == "slide":
            if self._best_x is None or self._pending_coords is not None:
                raise RuntimeError("invalid slide state")
            raise RuntimeError("slide candidate should have been prepared internally")

        if self._phase == "polish":
            if self._best_x is None or self.remaining <= 0:
                self._phase = "done"
                raise RuntimeError("optimization session is complete")
            span = self.hi - self.lo
            step = self.radius_schedule[-1] * span * max(0.25, 0.5 ** max(0, self._generation - 1))
            points = []
            for axis in range(self.dimension):
                for sign in (-1.0, 1.0):
                    if len(points) >= self.remaining:
                        break
                    cand = self._best_x.copy()
                    cand[axis] = np.clip(cand[axis] + sign * step, self.lo, self.hi)
                    points.append(cand)
                if len(points) >= self.remaining:
                    break
            if not points:
                self._phase = "done"
                raise RuntimeError("optimization session is complete")
            return self._set_pending(np.asarray(points), "polish")

        raise RuntimeError(f"unknown session phase: {self._phase}")

    def tell(self, values: np.ndarray | list[float]) -> None:
        if self._pending is None or self._pending_kind is None:
            raise RuntimeError("ask() must be called before tell()")
        vals = np.asarray(values, dtype=float)
        if vals.shape != (len(self._pending),):
            raise ValueError("values must contain one scalar for each asked point")
        if not np.all(np.isfinite(vals)):
            raise ValueError("values must be finite")

        kind = self._pending_kind
        points = self._pending
        coords = self._pending_coords
        j = int(np.argmin(vals))
        if float(vals[j]) < self._best_fun:
            self._best_fun = float(vals[j])
            self._best_x = points[j].copy()

        self._evaluations += len(points)
        self._pending = None
        self._pending_kind = None
        self._pending_coords = None
        self._generation += 1

        if kind == "explore":
            if self._evaluations >= self.exploration_budget or self.remaining <= 0:
                self._phase = "spiral" if self.remaining > 0 else "done"
            return

        if kind == "spiral":
            self._last_spiral_values = vals.copy()
            if coords is not None and self._plane is not None and self.remaining > 0 and self._best_x is not None:
                grad, hessian = _fit_relief(coords, vals)
                step2 = None
                eig = np.linalg.eigvalsh(hessian)
                if np.all(eig > 1e-10):
                    try:
                        step2 = -np.linalg.solve(hessian + 1e-10 * np.eye(2), grad)
                    except np.linalg.LinAlgError:
                        step2 = None
                if step2 is None or not np.all(np.isfinite(step2)):
                    norm = float(np.linalg.norm(grad))
                    if norm > 1e-15:
                        radius = self.radius_schedule[self._level] * (self.hi - self.lo)
                        step2 = -radius * grad / norm
                if step2 is not None and np.all(np.isfinite(step2)):
                    radius = self.radius_schedule[self._level] * (self.hi - self.lo)
                    norm = float(np.linalg.norm(step2))
                    if norm > radius:
                        step2 = step2 * radius / (norm + 1e-15)
                    u, v = self._plane
                    cand = np.clip(self._best_x + step2[0] * u + step2[1] * v, self.lo, self.hi)
                    self._pending = cand.reshape(1, -1)
                    self._pending_kind = "slide-candidate"
                    self._pending_coords = None
                    self._phase = "slide-candidate"
                    return
            self._level += 1
            self._phase = "spiral" if self._level < len(self.radius_schedule) else "polish"
            return

        if kind == "slide-candidate":
            self._level += 1
            self._phase = "spiral" if self._level < len(self.radius_schedule) and self.remaining > 0 else "polish"
            return

        if kind == "polish":
            if self.remaining <= 0:
                self._phase = "done"
            return

    def pending_batch(self) -> AskBatch | None:
        """Return an internally prepared batch, used for slide candidates."""
        if self._pending is None:
            return None
        return AskBatch(self._pending.copy(), self._generation)

    def result(self) -> OptimizationResult:
        if self._best_x is None:
            raise RuntimeError("no observations have been supplied")
        return OptimizationResult(
            self._best_x.copy(),
            self._best_fun,
            self._evaluations,
            self.seed,
            "RO-Multires-AskTell-exp",
            status="success" if self.done else "running",
            diagnostics={
                "phase": self._phase,
                "level": self._level,
                "generation": self._generation,
                "remaining_budget": self.remaining,
                "protocol": "ask-tell-multires-experimental",
            },
        )
