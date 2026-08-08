"""Shared types and helpers for resolutive optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Objective = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class OptimizationResult:
    x: np.ndarray
    fun: float
    evaluations: int
    seed: int
    version: str


def validate_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    lo, hi = map(float, bounds)
    if not lo < hi:
        raise ValueError("bounds must satisfy lower < upper")
    return lo, hi
