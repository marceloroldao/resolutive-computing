"""Classical continuous black-box benchmark functions."""

from __future__ import annotations

import numpy as np


def ackley(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x * x)))
        - np.exp(np.mean(np.cos(2.0 * np.pi * x)))
        + np.e
        + 20.0
    )


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(10.0 * x.size + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


DEFAULT_BENCHMARKS = {
    "ackley": (ackley, (-32.768, 32.768)),
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-5.0, 10.0)),
}
