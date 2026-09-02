"""Stateful ask/tell port of the regime-selecting Resolutive hybrid."""
from __future__ import annotations

import numpy as np

from .optimization.common import OptimizationResult, validate_bounds
from .optimization.spiral_relief import _fit_relief, _orthonormal_plane
from .session import AskBatch
from .specialist_sessions import V2Session, V5Session
from .spiral_session import SpiralRefineSession


class HybridRegimeSession:
    """Incremental reconnaissance -> specialist -> geometry-gated refinement."""

    def __init__(self, *, dimension: int, bounds: tuple[float, float], budget: int = 6000,
                 seed: int = 0, reconnaissance_fraction: float = 0.04,
                 local_fraction: float = 0.14, ruggedness_threshold: float = 0.28,
                 anisotropy_threshold: float = 8.0) -> None:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        self.lo, self.hi = validate_bounds(bounds)
        if budget < 600:
            raise ValueError("budget must be >= 600")
        self.dimension = int(dimension)
        self.budget = int(budget)
        self.seed = int(seed)
        self.recon_budget = max(40, int(round(budget * reconnaissance_fraction)))
        self.local_budget = max(140, int(round(budget * local_fraction)))
        self.core_budget = budget - self.recon_budget - self.local_budget
        self.ruggedness_threshold = float(ruggedness_threshold)
        self.anisotropy_threshold = float(anisotropy_threshold)
        self.span = self.hi - self.lo

        self.phase = "recon"
        self.pending: np.ndarray | None = None
        self.pending_kind: str | None = None
        self.evaluations = 0
        self.best_x: np.ndarray | None = None
        self.best_fun = float("inf")
        self.ruggedness = 0.0
        self.core_name: str | None = None
        self.core = None
        self.local = None

        rng = np.random.default_rng(seed + 700001)
        self.recon_center = rng.uniform(self.lo, self.hi, self.dimension)
        directions = max(4, min(16, (self.recon_budget - 1) // 3))
        points = [self.recon_center]
        self.recon_triplets: list[tuple[int, int, int]] = []
        for _ in range(directions):
            d = rng.normal(size=self.dimension)
            d /= np.linalg.norm(d) + 1e-15
            idx0 = len(points)
            for frac in (0.04, 0.10, 0.20):
                points.append(np.clip(self.recon_center + frac * self.span * d, self.lo, self.hi))
            self.recon_triplets.append((idx0, idx0 + 1, idx0 + 2))
        self.recon_points = np.asarray(points, dtype=float)
        self.recon_vals: np.ndarray | None = None

    @property
    def remaining(self) -> int:
        return self.budget - self.evaluations

    @property
    def done(self) -> bool:
        return self.phase == "done" or self.remaining <= 0

    def _emit(self, points: np.ndarray, kind: str) -> AskBatch:
        self.pending = np.asarray(points, dtype=float)
        self.pending_kind = kind
        return AskBatch(self.pending.copy(), self.evaluations)

    def _advance_after_core(self) -> AskBatch:
        r = self.core.result()
        if r.fun < self.best_fun:
            self.best_fun, self.best_x = float(r.fun), r.x.copy()
        self.phase = "gate" if self.remaining >= 30 else "done"
        if self.done:
            raise RuntimeError("optimization session is complete")
        return self.ask()

    def ask(self) -> AskBatch:
        if self.pending is not None:
            raise RuntimeError("tell() must be called before the next ask()")
        if self.done:
            raise RuntimeError("optimization session is complete")
        if self.phase == "recon":
            return self._emit(self.recon_points, "recon")
        if self.phase == "core":
            if self.core.done:
                return self._advance_after_core()
            try:
                return self.core.ask()
            except RuntimeError as exc:
                # Specialist sessions may discover at ask-time that the
                # remaining private core budget cannot fit another full
                # generation. Treat that as normal phase completion rather
                # than leaking the specialist's terminal exception outward.
                if self.core.done and str(exc) == "optimization session is complete":
                    return self._advance_after_core()
                raise
        if self.phase == "gate":
            rng = np.random.default_rng(self.seed + 700019)
            u, v = _orthonormal_plane(rng, self.dimension)
            self.gate_plane = (u, v)
            probe_n = min(18, max(8, self.remaining // 5))
            idx = np.arange(1, probe_n + 1, dtype=float)
            theta = np.pi * (3.0 - np.sqrt(5.0)) * idx * 2.5
            radial = 0.025 * self.span * np.sqrt(idx / probe_n)
            self.gate_coords = np.column_stack((radial * np.cos(theta), radial * np.sin(theta)))
            pts = np.clip(self.best_x + self.gate_coords[:, :1] * u + self.gate_coords[:, 1:] * v,
                          self.lo, self.hi)
            return self._emit(pts, "gate")
        if self.phase == "local":
            if self.local.done:
                r = self.local.result()
                if r.fun < self.best_fun:
                    self.best_fun, self.best_x = float(r.fun), r.x.copy()
                self.phase = "done"
                raise RuntimeError("optimization session is complete")
            return self.local.ask()
        if self.phase == "collapse":
            step = getattr(self, "collapse_step", 0.025 * self.span)
            pts = []
            for axis in range(self.dimension):
                for sign in (-1.0, 1.0):
                    if len(pts) >= self.remaining:
                        break
                    p = self.best_x.copy()
                    p[axis] = np.clip(p[axis] + sign * step, self.lo, self.hi)
                    pts.append(p)
                if len(pts) >= self.remaining:
                    break
            if not pts:
                self.phase = "done"
                raise RuntimeError("optimization session is complete")
            return self._emit(np.asarray(pts), "collapse")
        raise RuntimeError(f"unknown phase: {self.phase}")

    def tell(self, values) -> None:
        if self.phase == "core":
            before = self.core.evaluations
            self.core.tell(values)
            self.evaluations += self.core.evaluations - before
            r = self.core.result()
            if r.fun < self.best_fun:
                self.best_fun, self.best_x = float(r.fun), r.x.copy()
            if self.core.done:
                self.phase = "gate" if self.remaining >= 30 else "done"
            return
        if self.phase == "local":
            before = self.local.used
            self.local.tell(values)
            self.evaluations += self.local.used - before
            r = self.local.result()
            if r.fun < self.best_fun:
                self.best_fun, self.best_x = float(r.fun), r.x.copy()
            if self.local.done:
                self.phase = "done"
            return

        if self.pending is None or self.pending_kind is None:
            raise RuntimeError("ask() must be called before tell()")
        vals = np.asarray(values, dtype=float)
        if vals.shape != (len(self.pending),) or not np.all(np.isfinite(vals)):
            raise ValueError("values must contain one finite scalar for each asked point")
        kind, points = self.pending_kind, self.pending.copy()
        self.evaluations += len(points)
        self.pending = None
        self.pending_kind = None

        j = int(np.argmin(vals))
        if float(vals[j]) < self.best_fun:
            self.best_fun, self.best_x = float(vals[j]), points[j].copy()

        if kind == "recon":
            self.recon_vals = vals.copy()
            sign_changes = 0
            valid = 0
            for a, b, c in self.recon_triplets:
                d1, d2 = vals[b] - vals[a], vals[c] - vals[b]
                if np.isfinite(d1) and np.isfinite(d2):
                    valid += 1
                    if d1 * d2 < 0.0:
                        sign_changes += 1
            self.ruggedness = sign_changes / max(1, valid)
            if self.ruggedness >= self.ruggedness_threshold:
                self.core_name = "V2"
                self.core = V2Session(dimension=self.dimension, bounds=(self.lo, self.hi),
                                      budget=self.core_budget, seed=self.seed)
            else:
                self.core_name = "V5"
                self.core = V5Session(dimension=self.dimension, bounds=(self.lo, self.hi),
                                      budget=self.core_budget, seed=self.seed, use_collapse=False)
            self.phase = "core"
            return

        if kind == "gate":
            grad, hessian = _fit_relief(self.gate_coords, vals)
            eig = np.linalg.eigvalsh(hessian)
            positive = bool(np.all(eig > 1e-10))
            anis = float((eig[-1] + 1e-15) / (eig[0] + 1e-15)) if positive else 0.0
            self.anisotropy = anis
            if positive and anis >= self.anisotropy_threshold and self.remaining >= 20:
                self.local = SpiralRefineSession(start=self.best_x, bounds=(self.lo, self.hi),
                                                 budget=self.remaining, seed=self.seed + 700027,
                                                 spiral_points=18, turns=2.5,
                                                 radius_fraction=0.03)
                self.phase = "local"
            else:
                self.collapse_step = 0.025 * self.span
                self.phase = "collapse"
            return

        if kind == "collapse":
            improved = float(np.min(vals)) < self.best_fun
            if improved:
                k = int(np.argmin(vals))
                self.best_fun, self.best_x = float(vals[k]), points[k].copy()
            else:
                self.collapse_step *= 0.5
            if self.remaining < 2 * self.dimension or self.collapse_step <= 1e-11 * self.span:
                self.phase = "done"
            return

    def result(self) -> OptimizationResult:
        if self.best_x is None:
            raise RuntimeError("no observations have been supplied")
        return OptimizationResult(
            self.best_x.copy(), self.best_fun, self.evaluations, self.seed,
            f"RO-Hybrid-AskTell[{self.core_name};rug={self.ruggedness:.3f}]",
            status="success" if self.done else "running",
            diagnostics={"phase": self.phase, "remaining_budget": self.remaining,
                         "ruggedness": self.ruggedness, "core": self.core_name,
                         "anisotropy": getattr(self, "anisotropy", None),
                         "protocol": "ask-tell-hybrid"},
        )
