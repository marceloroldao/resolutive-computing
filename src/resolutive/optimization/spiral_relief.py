"""Experimental spiral-relief black-box optimizer.

The method samples a 2-D spiral embedded in the n-D search space, fits a local
quadratic relief to objective values, and then slides the center downhill on
that inferred relief. It can operate globally from a random point or locally
from a supplied incumbent solution.
"""
from __future__ import annotations

import numpy as np

from .common import Objective, OptimizationResult, validate_bounds


def _orthonormal_plane(rng: np.random.Generator, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    u = rng.normal(size=dimension)
    u /= np.linalg.norm(u) + 1e-15
    v = rng.normal(size=dimension)
    v -= np.dot(v, u) * u
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        axis = int(np.argmin(np.abs(u)))
        v = np.zeros(dimension)
        v[axis] = 1.0
        v -= np.dot(v, u) * u
        norm = np.linalg.norm(v)
    v /= norm + 1e-15
    return u, v


def _fit_relief(coords: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit z=c0+c1*a+c2*b+c3*a^2+c4*a*b+c5*b^2."""
    a = coords[:, 0]
    b = coords[:, 1]
    design = np.column_stack((np.ones(len(coords)), a, b, a * a, a * b, b * b))
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    gradient = np.array([coef[1], coef[2]], dtype=float)
    hessian = np.array([[2.0 * coef[3], coef[4]], [coef[4], 2.0 * coef[5]]], dtype=float)
    return gradient, hessian


def _slide_vector(gradient: np.ndarray, hessian: np.ndarray, radius: float) -> np.ndarray:
    """Choose a bounded downhill displacement on the fitted relief."""
    gnorm = float(np.linalg.norm(gradient))
    if not np.isfinite(gnorm) or gnorm < 1e-15:
        return np.zeros(2)

    eigvals = np.linalg.eigvalsh(hessian)
    if np.all(eigvals > 1e-10):
        damping = max(1e-8, 0.05 * float(np.max(eigvals)))
        try:
            step = -np.linalg.solve(hessian + damping * np.eye(2), gradient)
        except np.linalg.LinAlgError:
            step = -gradient / gnorm
    else:
        step = -gradient / gnorm

    snorm = float(np.linalg.norm(step))
    max_step = 0.85 * radius
    if snorm > max_step:
        step *= max_step / (snorm + 1e-15)
    return step


class SpiralReliefOptimizer:
    """Map-and-slide optimizer based on local spiral sampling."""

    def __init__(self, spiral_points: int = 18, turns: float = 2.5) -> None:
        if spiral_points < 8:
            raise ValueError("spiral_points must be >= 8")
        if turns <= 0:
            raise ValueError("turns must be > 0")
        self.spiral_points = spiral_points
        self.turns = float(turns)

    def _run_from(self, objective: Objective, *, start: np.ndarray,
                  bounds: tuple[float, float], budget: int, seed: int,
                  radius_fraction: float, version: str) -> OptimizationResult:
        lo, hi = validate_bounds(bounds)
        dimension = int(len(start))
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        if budget <= self.spiral_points + 1:
            raise ValueError("budget too small for one spiral cycle")
        if not (0.0 < radius_fraction <= 0.5):
            raise ValueError("radius_fraction must be in (0, 0.5]")

        rng = np.random.default_rng(seed)
        span = hi - lo
        center = np.clip(np.asarray(start, dtype=float), lo, hi).copy()
        center_value = float(objective(center))
        best = center.copy()
        best_value = center_value
        used = 1

        radius = radius_fraction * span
        min_radius = 1e-7 * span
        stall = 0
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        while used + self.spiral_points + 1 <= budget and radius >= min_radius:
            u, v = _orthonormal_plane(rng, dimension)
            idx = np.arange(1, self.spiral_points + 1, dtype=float)
            radial = radius * np.sqrt(idx / self.spiral_points)
            theta = golden_angle * idx * self.turns
            coords = np.column_stack((radial * np.cos(theta), radial * np.sin(theta)))

            points = center + coords[:, :1] * u + coords[:, 1:] * v
            points = np.clip(points, lo, hi)
            values = np.array([float(objective(p)) for p in points])
            used += self.spiral_points

            local_i = int(np.argmin(values))
            mapped_value = float(values[local_i])
            mapped_point = points[local_i].copy()
            if mapped_value < best_value:
                best = mapped_point.copy()
                best_value = mapped_value

            gradient, hessian = _fit_relief(coords, values)
            slide2 = _slide_vector(gradient, hessian, radius)
            candidate = np.clip(center + slide2[0] * u + slide2[1] * v, lo, hi)
            candidate_value = float(objective(candidate))
            used += 1
            if candidate_value < best_value:
                best = candidate.copy()
                best_value = candidate_value

            if min(mapped_value, candidate_value) < center_value:
                if candidate_value <= mapped_value:
                    center, center_value = candidate, candidate_value
                else:
                    center, center_value = mapped_point, mapped_value
                # Local refinement should not re-expand aggressively.
                radius *= 0.92
                stall = 0
            else:
                stall += 1
                radius *= 0.72 if stall >= 2 else 0.90
                if stall >= 2:
                    stall = 0

        return OptimizationResult(best, best_value, used, seed, version)

    def refine(self, objective: Objective, *, start: np.ndarray,
               bounds: tuple[float, float], budget: int = 600,
               seed: int = 0, radius_fraction: float = 0.04) -> OptimizationResult:
        """Refine an incumbent point using a small local spiral relief."""
        return self._run_from(
            objective, start=start, bounds=bounds, budget=budget, seed=seed,
            radius_fraction=radius_fraction, version="SpiralRelief-local",
        )

    def minimize(self, objective: Objective, *, dimension: int,
                 bounds: tuple[float, float], budget: int = 6000,
                 seed: int = 0) -> OptimizationResult:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        lo, hi = validate_bounds(bounds)
        rng = np.random.default_rng(seed)
        start = rng.uniform(lo, hi, dimension)
        return self._run_from(
            objective, start=start, bounds=bounds, budget=budget, seed=seed + 1,
            radius_fraction=0.24, version="SpiralRelief-exp",
        )
