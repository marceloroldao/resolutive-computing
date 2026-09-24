"""Stateful ask/tell ports of Resolutive V2 and V5.

These sessions preserve the population dynamics of the monolithic engines while
ensuring every external evaluation is emitted exclusively through ``ask()``.
"""
from __future__ import annotations

import numpy as np

from .optimization.common import OptimizationResult, validate_bounds
from .session import AskBatch


class _BasePopulationSession:
    def __init__(self, *, dimension: int, bounds: tuple[float, float], budget: int,
                 seed: int, population: int, collapse_budget: int, version: str) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        if budget <= population:
            raise ValueError("budget must exceed population")
        self.dimension = int(dimension)
        self.lo, self.hi = validate_bounds(bounds)
        self.budget = int(budget)
        self.seed = int(seed)
        self.population = int(population)
        self.collapse_budget = int(collapse_budget)
        self.version = version
        self.span = self.hi - self.lo
        self.rng = np.random.default_rng(seed)
        self._evaluations = 0
        self._generation = 0
        self._phase = "init"
        self._pending: np.ndarray | None = None
        self._pending_kind: str | None = None
        self.x: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self.best_x: np.ndarray | None = None
        self.best_fun = float("inf")
        self._collapse_step = 0.05 * self.span
        self._collapse_point: np.ndarray | None = None
        self._collapse_improved = False

    @property
    def evaluations(self) -> int:
        return self._evaluations

    @property
    def remaining(self) -> int:
        return self.budget - self._evaluations

    @property
    def done(self) -> bool:
        return self._phase == "done" or self.remaining <= 0

    def _set_pending(self, points: np.ndarray, kind: str) -> AskBatch:
        self._pending = np.asarray(points, dtype=float)
        self._pending_kind = kind
        return AskBatch(self._pending.copy(), self._generation)

    def _update_best_from_population(self) -> None:
        assert self.x is not None and self.values is not None
        i = int(np.argmin(self.values))
        if float(self.values[i]) < self.best_fun:
            self.best_fun = float(self.values[i])
            self.best_x = self.x[i].copy()

    def _collapse_points(self) -> np.ndarray:
        assert self._collapse_point is not None
        pts: list[np.ndarray] = []
        for axis in range(self.dimension):
            for sign in (-1.0, 1.0):
                if len(pts) >= self.remaining:
                    break
                c = self._collapse_point.copy()
                c[axis] = np.clip(c[axis] + sign * self._collapse_step, self.lo, self.hi)
                pts.append(c)
            if len(pts) >= self.remaining:
                break
        return np.asarray(pts, dtype=float)

    def _handle_collapse(self, values: np.ndarray) -> None:
        assert self._pending is not None and self._collapse_point is not None
        improved = False
        for p, v in zip(self._pending, values):
            if float(v) < self.best_fun:
                self.best_fun = float(v)
                self.best_x = p.copy()
                self._collapse_point = p.copy()
                improved = True
        if not improved:
            self._collapse_step *= 0.5
        if self.remaining < 2 * self.dimension or self._collapse_step <= 1e-10 * self.span:
            self._phase = "done"

    def result(self) -> OptimizationResult:
        if self.best_x is None:
            raise RuntimeError("no observations have been supplied")
        return OptimizationResult(
            self.best_x.copy(), self.best_fun, self._evaluations, self.seed,
            self.version,
            status="success" if self.done else "running",
            diagnostics={
                "phase": self._phase,
                "generation": self._generation,
                "remaining_budget": self.remaining,
                "protocol": "ask-tell-specialist",
            },
        )


class V2Session(_BasePopulationSession):
    def __init__(self, *, dimension: int, bounds: tuple[float, float], budget: int,
                 seed: int = 0, population: int = 36) -> None:
        super().__init__(dimension=dimension, bounds=bounds, budget=budget, seed=seed,
                         population=population, collapse_budget=max(200, budget // 20),
                         version="RO-V2-AskTell")
        self.stall = 0
        self.golden_angle = 2.0 * np.pi * (1.0 - 1.0 / ((1.0 + np.sqrt(5.0)) / 2.0))

    def ask(self) -> AskBatch:
        if self._pending is not None:
            raise RuntimeError("tell() must be called before the next ask()")
        if self.done:
            raise RuntimeError("optimization session is complete")
        if self._phase == "init":
            return self._set_pending(
                self.rng.uniform(self.lo, self.hi, (self.population, self.dimension)), "init")
        if self._phase == "core":
            assert self.x is not None and self.values is not None and self.best_x is not None
            if self._evaluations + self.population > self.budget - self.collapse_budget:
                self._phase = "collapse"
                self._collapse_point = self.best_x.copy()
                return self.ask()
            self._generation += 1
            order = np.argsort(self.values)
            elite = self.x[order[: max(5, self.population // 4)]]
            covariance = np.cov(elite, rowvar=False) + np.eye(self.dimension) * 1e-10
            spread = np.sqrt(np.trace(covariance)) / self.span
            coherence = np.clip(
                0.5 * (self._evaluations / self.budget)
                + 0.5 * (1.0 - np.clip(spread / 0.25, 0.0, 1.0)), 0.0, 1.0)
            vertical = 0.28 * self.span * (1.0 - coherence) ** 2.2
            horizontal = 0.18 + 0.72 * coherence
            proposals = []
            for i, row in enumerate(self.x):
                target = elite[self.rng.integers(len(elite))]
                direction = horizontal * (0.58 * (self.best_x - row) + 0.42 * (target - row))
                if self.dimension > 1:
                    a, b = self.rng.choice(self.dimension, 2, replace=False)
                    angle = self.golden_angle * (self._generation + i + 1)
                    va, vb = direction[a], direction[b]
                    direction[a] = np.cos(angle) * va - np.sin(angle) * vb
                    direction[b] = np.sin(angle) * va + np.cos(angle) * vb
                noise = self.rng.multivariate_normal(np.zeros(self.dimension), covariance)
                noise /= np.linalg.norm(noise) + 1e-12
                proposals.append(np.clip(row + direction + vertical * noise, self.lo, self.hi))
            return self._set_pending(np.asarray(proposals), "core")
        if self._phase == "reseed":
            assert self.values is not None
            worst = np.argsort(self.values)[-max(2, self.population // 8):]
            self._reseed_indices = worst
            return self._set_pending(
                self.rng.uniform(self.lo, self.hi, (len(worst), self.dimension)), "reseed")
        if self._phase == "collapse":
            assert self._collapse_point is not None
            pts = self._collapse_points()
            if len(pts) == 0:
                self._phase = "done"
                raise RuntimeError("optimization session is complete")
            return self._set_pending(pts, "collapse")
        raise RuntimeError(f"unknown phase {self._phase}")

    def tell(self, values) -> None:
        if self._pending is None or self._pending_kind is None:
            raise RuntimeError("ask() must be called before tell()")
        vals = np.asarray(values, dtype=float)
        if vals.shape != (len(self._pending),) or not np.all(np.isfinite(vals)):
            raise ValueError("values must contain one finite scalar for each asked point")
        kind, points = self._pending_kind, self._pending.copy()
        self._evaluations += len(points)
        self._pending = None; self._pending_kind = None
        if kind == "init":
            self.x, self.values = points, vals.copy()
            self._update_best_from_population(); self._phase = "core"; return
        if kind == "core":
            assert self.x is not None and self.values is not None
            improved = vals < self.values
            self.x = np.where(improved[:, None], points, self.x)
            self.values = np.where(improved, vals, self.values)
            old = self.best_fun; self._update_best_from_population()
            self.stall = 0 if self.best_fun < old else self.stall + 1
            if self.stall >= 14 and self._evaluations < self.budget - self.collapse_budget:
                self._phase = "reseed"
            return
        if kind == "reseed":
            assert self.x is not None and self.values is not None
            idx = self._reseed_indices
            self.x[idx], self.values[idx] = points, vals
            self._update_best_from_population(); self.stall = 0; self._phase = "core"; return
        if kind == "collapse":
            self._pending = points
            self._handle_collapse(vals)
            self._pending = None
            if self._phase != "done": self._phase = "collapse"


class V5Session(_BasePopulationSession):
    def __init__(self, *, dimension: int, bounds: tuple[float, float], budget: int,
                 seed: int = 0, population: int = 40, use_collapse: bool = False) -> None:
        collapse_budget = max(300, budget // 15) if use_collapse else 0
        super().__init__(dimension=dimension, bounds=bounds, budget=budget, seed=seed,
                         population=population, collapse_budget=collapse_budget,
                         version="RO-V5-AskTell")
        self.use_collapse = bool(use_collapse)
        self.stall = 0
        self.escape = False
        self.escape_left = 0
        self.history: list[float] = []
        self.bad_regions: list[tuple[np.ndarray, float, float]] = []
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        self.golden_angle = 2.0 * np.pi * (1.0 - 1.0 / phi)

    def ask(self) -> AskBatch:
        if self._pending is not None:
            raise RuntimeError("tell() must be called before the next ask()")
        if self.done:
            raise RuntimeError("optimization session is complete")
        if self._phase == "init":
            return self._set_pending(
                self.rng.uniform(self.lo, self.hi, (self.population, self.dimension)), "init")
        if self._phase == "core":
            assert self.x is not None and self.values is not None and self.best_x is not None
            if self._evaluations + self.population > self.budget - self.collapse_budget:
                if self.use_collapse:
                    self._phase = "collapse"; self._collapse_point = self.best_x.copy(); return self.ask()
                self._phase = "done"; raise RuntimeError("optimization session is complete")
            self._generation += 1
            order = np.argsort(self.values)
            elite = self.x[order[: max(5, self.population // 4)]]
            covariance = np.cov(elite, rowvar=False) + np.eye(self.dimension) * 1e-10
            eigvals, eigvecs = np.linalg.eigh(covariance)
            spread = np.sqrt(np.trace(covariance)) / self.span
            anisotropy = (eigvals[-1] + 1e-15) / (eigvals[0] + 1e-15)
            coherence = np.clip(
                0.5 * (self._evaluations / self.budget)
                + 0.5 * (1.0 - np.clip(spread / 0.25, 0.0, 1.0)), 0.0, 1.0)
            self.history = (self.history + [self.best_fun])[-12:]
            rel = 0.0
            if len(self.history) >= 8:
                rel = (self.history[-8] - self.history[-1]) / (abs(self.history[-8]) + 1e-12)
            if not self.escape and self.stall >= 10 and rel < 1e-5:
                self.escape = True; self.escape_left = 8
            proposals = []
            if not self.escape:
                vertical = 0.28 * self.span * (1.0 - coherence) ** 2.2
                horizontal = 0.18 + 0.72 * coherence
                try:
                    chol = np.linalg.cholesky(covariance)
                except np.linalg.LinAlgError:
                    chol = np.eye(self.dimension) * max(np.std(elite), 1e-8)
                for i, row in enumerate(self.x):
                    target = elite[self.rng.integers(len(elite))]
                    direction = horizontal * (0.58 * (self.best_x - row) + 0.42 * (target - row))
                    if self.dimension > 1:
                        a, b = self.rng.choice(self.dimension, 2, replace=False)
                        angle = self.golden_angle * (self._generation + i + 1)
                        va, vb = direction[a], direction[b]
                        direction[a] = np.cos(angle) * va - np.sin(angle) * vb
                        direction[b] = np.sin(angle) * va + np.cos(angle) * vb
                    repulsion = np.zeros(self.dimension)
                    for center, radius, weight in self.bad_regions:
                        delta = row - center
                        repulsion += weight * np.exp(-np.dot(delta, delta)/(2*radius**2+1e-12)) * delta/(radius**2+1e-12)
                    noise = chol @ self.rng.normal(size=self.dimension)
                    noise /= np.linalg.norm(noise) + 1e-12
                    proposals.append(np.clip(row + direction + vertical * noise + 0.05 * repulsion, self.lo, self.hi))
            else:
                for ids in np.array_split(np.arange(self.population), 4):
                    local_order = ids[np.argsort(self.values[ids])]
                    local_elite = self.x[local_order[: max(2, len(local_order)//3)]]
                    local_best = local_elite[0]
                    for idx in ids:
                        row = self.x[idx]; target = local_elite[self.rng.integers(len(local_elite))]
                        noise = self.rng.normal(size=self.dimension); noise /= np.linalg.norm(noise)+1e-12
                        step = 0.24*(local_best-row)+0.20*(target-row)+0.18*self.span*noise
                        if anisotropy > 80.0:
                            step += 0.08*self.span*(1.0 if self.rng.random()<0.5 else -1.0)*eigvecs[:, -1]
                        proposals.append(np.clip(row + step, self.lo, self.hi))
            self._coherence = coherence
            return self._set_pending(np.asarray(proposals), "core")
        if self._phase == "reanchor":
            assert self.values is not None and self.best_x is not None
            worst = np.argsort(self.values)[-max(2, self.population // 8):]
            self._reanchor_indices = worst
            pts = np.clip(self.best_x + self.rng.normal(0.0, 0.03*self.span, (len(worst), self.dimension)), self.lo, self.hi)
            return self._set_pending(pts, "reanchor")
        if self._phase == "collapse":
            pts = self._collapse_points()
            if len(pts)==0:
                self._phase="done"; raise RuntimeError("optimization session is complete")
            return self._set_pending(pts, "collapse")
        raise RuntimeError(f"unknown phase {self._phase}")

    def tell(self, values) -> None:
        if self._pending is None or self._pending_kind is None:
            raise RuntimeError("ask() must be called before tell()")
        vals = np.asarray(values, dtype=float)
        if vals.shape != (len(self._pending),) or not np.all(np.isfinite(vals)):
            raise ValueError("values must contain one finite scalar for each asked point")
        kind, points = self._pending_kind, self._pending.copy()
        self._evaluations += len(points)
        self._pending=None; self._pending_kind=None
        if kind=="init":
            self.x, self.values = points, vals.copy(); self._update_best_from_population(); self._phase="core"; return
        if kind=="core":
            assert self.x is not None and self.values is not None
            improved = vals < self.values
            if len(np.where(~improved)[0]) and self._generation % 4 == 0:
                rejected = np.where(~improved)[0]
                worst = rejected[np.argmax(vals[rejected])]
                radius = max(0.02*self.span, 0.12*self.span*(1.0-self._coherence))
                self.bad_regions.append((points[worst].copy(), radius, 1.0))
                self.bad_regions = [(c,r,w*0.94) for c,r,w in self.bad_regions if w*0.94>0.15][-24:]
            self.x = np.where(improved[:,None], points, self.x)
            self.values = np.where(improved, vals, self.values)
            old=self.best_fun; self._update_best_from_population(); self.stall=0 if self.best_fun<old else self.stall+1
            if self.escape:
                self.escape_left -= 1
                if self.escape_left <= 0:
                    self.escape=False; self.stall=0; self._phase="reanchor"; return
            self._phase="core"; return
        if kind=="reanchor":
            assert self.x is not None and self.values is not None
            idx=self._reanchor_indices; self.x[idx], self.values[idx]=points, vals; self._update_best_from_population(); self._phase="core"; return
        if kind=="collapse":
            self._pending=points; self._handle_collapse(vals); self._pending=None
            if self._phase!="done": self._phase="collapse"
