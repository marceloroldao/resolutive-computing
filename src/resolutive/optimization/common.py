"""Shared types and helpers for resolutive optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

Objective = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class OptimizationResult:
    """Stable public result contract for optimization calls.

    The original positional fields are preserved for backwards compatibility.
    New API-facing fields are appended with defaults so existing engines do not
    need to change their constructors immediately.
    """

    x: np.ndarray
    fun: float
    evaluations: int
    seed: int
    version: str
    status: str = "success"
    diagnostics: dict[str, Any] | None = None

    @property
    def optimizer(self) -> str:
        """Public alias for the internal historical ``version`` field."""
        return self.version


def validate_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    lo, hi = map(float, bounds)
    if not lo < hi:
        raise ValueError("bounds must satisfy lower < upper")
    return lo, hi
