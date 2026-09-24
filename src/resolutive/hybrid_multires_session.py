"""Stateful ask/tell port of the Hybrid MultiResolution optimizer.

The core phase delegates to :class:`HybridRegimeSession`, preserving the
reconnaissance/specialist dynamics before the progressively finer local spiral
maps.  All objective evaluations are emitted only by ``ask()``.
"""
from __future__ import annotations

import numpy as np

from .hybrid_session import HybridRegimeSession
from .optimization.common import OptimizationResult, validate_bounds
from .optimization.spiral_relief import _fit_relief, _orthonormal_plane
from .session import AskBatch


class HybridMultiResolutionSession:
    """Incremental Hybrid -> multiresolution -> coordinate-polish optimizer."""

    def __init__(
        self,
        *,
        dimension: int,
        bounds: tuple[float, float],
        budget: int = 6000,
        seed: int = 0,
        local_fraction: float = 0.20,
        radius_schedule: tuple[float, ...] = (0.04, 0.01, 0.0025, 0.000625),
        points_per_level: int = 18,
        turns: float = 2.5,
    ) -> None:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        if budget < 800:
            raise ValueError("budget must be >= 800")
        if not 0.12 <= local_fraction <= 0.35:
            raise ValueError("local_fraction must be in [0.12, 0.35]")
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
        self.radius_schedule = tuple(float(r) for r in radius_schedule)
        self.points_per_level = int(points_per_level)
        self.turns = float(turns)
        self.span = self.hi - self.lo

        self.local_budget = max(220, int(round(self.budget * local_fraction)))
        self.core_budget = self.budget - self.local_budget
        self.core = HybridRegimeSession(
            dimension=self.dimension,
            bounds=(self.lo, self.hi),
            budget=self.core_budget,
            seed=self.seed,
            local_fraction=0.10,
        )

        self.phase = "core"
        self.level = 0
        self.evaluations = 0
        self.best_x: np.ndarray | None = None
        self.best_fun = float("inf")
        self.pending: np.ndarray | None = None
        self.pending_kind: str | None = None
        self.pending_coords: np.ndarray | None = None
        self.plane: tuple[np.ndarray, np.ndarray] | None = None
        self.prepared_slide: np.ndarray | None = None
        self.polish_step = self.radius_schedule[-1] * self.span

    @property
    def remaining(self) -> int:
        return self.budget - self.evaluations

    @property
    def done(self) -> bool:
        return self.phase == "done" or self.remaining <= 0

    def _sync_core_result(self) -> None:
        result = self.core.result()
        if result.fun < self.best_fun:
            self.best_fun = float(result.fun)
            self.best_x = result.x.copy()

    def _finish_core(self) -> None:
        self._sync_core_result()
        self.phase = "spiral" if self.remaining >= 10 else "done"

    def _emit(self, points: np.ndarray, kind: str,
              coords: np.ndarray | None = None) -> AskBatch:
        self.pending = np.asarray(points, dtype=float)
        self.pending_kind = kind
        self.pending_coords = None if coords is None else np.asarray(coords, dtype=float)
        return AskBatch(self.pending.copy(), self.evaluations)

    def ask(self) -> AskBatch:
        if self.pending is not None:
            raise RuntimeError("tell() must be called before the next ask()")
        if self.done:
            raise RuntimeError("optimization session is complete")

        if self.phase == "core":
            if self.core.done:
                self._finish_core()
                return self.ask()
            try:
                return self.core.ask()
            except RuntimeError as exc:
                if "complete" not in str(exc):
                    raise
                self._finish_core()
                return self.ask()

        if self.phase == "spiral":
            if self.best_x is None:
                self.phase = "done"
                raise RuntimeError("optimization session is complete")
            if self.level >= len(self.radius_schedule) or self.remaining < 10:
                self.phase = "polish"
                return self.ask()

            levels_left = len(self.radius_schedule) - self.level
            level_budget = max(8, self.remaining // levels_left)
            n = min(self.points_per_level, max(8, level_budget - 2))
            if n + 1 > self.remaining:
                n = self.remaining - 1
            if n < 8:
                self.phase = "polish"
                return self.ask()

            rng = np.random.default_rng(self.seed + 810001 + self.level * 9973)
            u, v = _orthonormal_plane(rng, self.dimension)
            self.plane = (u, v)
            idx = np.arange(1, n + 1, dtype=float)
            radius_fraction = self.radius_schedule[self.level]
            radial = radius_fraction * self.span * np.sqrt(idx / n)
            theta = np.pi * (3.0 - np.sqrt(5.0)) * idx * self.turns
            coords = np.column_stack((radial * np.cos(theta), radial * np.sin(theta)))
            points = np.clip(
                self.best_x + coords[:, :1] * u + coords[:, 1:] * v,
                self.lo,
                self.hi,
            )
            return self._emit(points, "spiral", coords)

        if self.phase == "slide":
            if self.prepared_slide is None:
                self.level += 1
                self.phase = "spiral"
                return self.ask()
            point = self.prepared_slide.reshape(1, -1)
            self.prepared_slide = None
            return self._emit(point, "slide")

        if self.phase == "polish":
            if self.best_x is None or self.remaining < 2 * self.dimension:
                self.phase = "done"
                raise RuntimeError("optimization session is complete")
            points: list[np.ndarray] = []
            for axis in range(self.dimension):
                for sign in (-1.0, 1.0):
                    cand = self.best_x.copy()
                    cand[axis] = np.clip(
                        cand[axis] + sign * self.polish_step, self.lo, self.hi)
                    points.append(cand)
            return self._emit(np.asarray(points, dtype=float), "polish")

        raise RuntimeError(f"unknown phase: {self.phase}")

    def tell(self, values) -> None:
        if self.phase == "core":
            before = self.core.evaluations
            self.core.tell(values)
            self.evaluations += self.core.evaluations - before
            self._sync_core_result()
            if self.core.done:
                self._finish_core()
            return

        if self.pending is None or self.pending_kind is None:
            raise RuntimeError("ask() must be called before tell()")
        vals = np.asarray(values, dtype=float)
        if vals.shape != (len(self.pending),):
            raise ValueError("values must contain one scalar for each asked point")
        if not np.all(np.isfinite(vals)):
            raise ValueError("values must be finite")

        kind = self.pending_kind
        points = self.pending.copy()
        coords = None if self.pending_coords is None else self.pending_coords.copy()
        self.evaluations += len(points)
        self.pending = None
        self.pending_kind = None
        self.pending_coords = None

        j = int(np.argmin(vals))
        if float(vals[j]) < self.best_fun:
            self.best_fun = float(vals[j])
            self.best_x = points[j].copy()

        if self.remaining <= 0:
            self.phase = "done"
            return

        if kind == "spiral":
            self.prepared_slide = None
            if coords is not None and self.plane is not None and self.best_x is not None:
                grad, hessian = _fit_relief(coords, vals)
                eig = np.linalg.eigvalsh(hessian)
                step2 = None
                if np.all(eig > 1e-10):
                    try:
                        step2 = -np.linalg.solve(
                            hessian + 1e-10 * np.eye(2), grad)
                    except np.linalg.LinAlgError:
                        step2 = None
                radius = self.radius_schedule[self.level] * self.span
                if step2 is None or not np.all(np.isfinite(step2)):
                    norm = float(np.linalg.norm(grad))
                    if norm > 1e-15:
                        step2 = -radius * grad / norm
                if step2 is not None and np.all(np.isfinite(step2)):
                    norm = float(np.linalg.norm(step2))
                    if norm > radius:
                        step2 = step2 * radius / (norm + 1e-15)
                    u, v = self.plane
                    self.prepared_slide = np.clip(
                        self.best_x + step2[0] * u + step2[1] * v,
                        self.lo,
                        self.hi,
                    )
            self.phase = "slide"
            return

        if kind == "slide":
            self.level += 1
            self.phase = "spiral"
            return

        if kind == "polish":
            # Match the monolithic policy: shrink only when a full axis pass
            # produced no improvement.
            if float(np.min(vals)) >= self.best_fun:
                self.polish_step *= 0.25
            if self.remaining < 2 * self.dimension or self.polish_step <= 1e-13 * self.span:
                self.phase = "done"
            else:
                self.phase = "polish"

    def result(self) -> OptimizationResult:
        if self.best_x is None:
            raise RuntimeError("no observations have been supplied")
        core_diag = self.core.result().diagnostics if self.core.best_x is not None else {}
        return OptimizationResult(
            self.best_x.copy(),
            self.best_fun,
            self.evaluations,
            self.seed,
            "RO-Hybrid-Multires-AskTell-exp",
            status="success" if self.done else "running",
            diagnostics={
                "phase": self.phase,
                "level": self.level,
                "remaining_budget": self.remaining,
                "core": core_diag,
                "protocol": "ask-tell-hybrid-multires",
            },
        )
