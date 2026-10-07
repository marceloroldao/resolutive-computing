"""Deterministic benchmark transformations for generalization testing.

These helpers create shifted/rotated instances and reproducible noisy objective
wrappers without changing optimizer implementations. They are intended to test
whether an optimizer generalizes beyond functions centered and aligned with the
coordinate axes.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Objective = Callable[[np.ndarray], float]


def orthogonal_matrix(dimension: int, seed: int) -> np.ndarray:
    """Return a deterministic orthogonal matrix with determinant +1."""
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q * signs
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def shift_vector(dimension: int, bounds: tuple[float, float], seed: int, fraction: float = 0.2) -> np.ndarray:
    """Create a deterministic interior shift vector."""
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    lo, hi = map(float, bounds)
    if not lo < hi:
        raise ValueError("bounds must satisfy lower < upper")
    if not 0.0 <= fraction < 0.5:
        raise ValueError("fraction must be in [0, 0.5)")
    rng = np.random.default_rng(seed)
    span = hi - lo
    limit = fraction * span
    return rng.uniform(-limit, limit, dimension)


def shifted_rotated(
    objective: Objective,
    *,
    dimension: int,
    bounds: tuple[float, float],
    seed: int,
    rotate: bool = True,
    shift_fraction: float = 0.2,
) -> tuple[Objective, np.ndarray, np.ndarray]:
    """Return f(Q @ (x - shift)), plus the shift and rotation used."""
    shift = shift_vector(dimension, bounds, seed, shift_fraction)
    rotation = orthogonal_matrix(dimension, seed + 1) if rotate else np.eye(dimension)

    def transformed(x: np.ndarray) -> float:
        z = rotation @ (np.asarray(x, dtype=float) - shift)
        return float(objective(z))

    return transformed, shift, rotation


class GaussianNoisyObjective:
    """Reproducible additive Gaussian measurement-noise wrapper.

    A fresh instance must be created for each optimizer run. Given the same
    query sequence and noise seed, the observations are exactly reproducible.
    """

    def __init__(self, objective: Objective, sigma: float, seed: int) -> None:
        if sigma < 0.0:
            raise ValueError("sigma must be >= 0")
        self.objective = objective
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> float:
        return float(self.objective(np.asarray(x, dtype=float)) + self.rng.normal(0.0, self.sigma))
