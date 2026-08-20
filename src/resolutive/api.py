"""Public Python API for Resolutive Computing.

This module is the compatibility boundary intended to converge toward v1.0.
Internal optimizer engines may evolve without requiring users to import their
implementation modules directly.
"""
from __future__ import annotations

from typing import Literal

from .optimization.common import Objective, OptimizationResult
from .optimization.hybrid_multires import ResolutiveHybridMultiResolution
from .optimization.hybrid_multires_robust import ResolutiveHybridMultiResolutionRobust
from .optimization.regime_router import ResolutiveRegimeRouter
from .optimization.v5 import ResolutiveV5
from .optimization.v6 import ResolutiveV6

Mode = Literal["auto", "v5", "v6", "multires", "robust"]


def optimize(
    objective: Objective,
    *,
    dimension: int,
    bounds: tuple[float, float],
    budget: int = 6000,
    seed: int = 0,
    mode: Mode = "auto",
) -> OptimizationResult:
    """Minimize a continuous black-box objective under an evaluation budget.

    Parameters
    ----------
    objective:
        Callable accepting a NumPy vector and returning a scalar objective.
    dimension:
        Number of decision variables.
    bounds:
        Shared ``(lower, upper)`` bound for every decision variable.
    budget:
        Maximum objective-evaluation budget.
    seed:
        Reproducibility seed.
    mode:
        ``auto`` delegates to the experimental explicit regime router;
        specialist modes select a specific engine.

    Notes
    -----
    The API shape is a v0.x candidate and is not yet guaranteed stable. The
    ``auto`` router remains experimental until its routing benchmarks pass.
    """
    engines = {
        "auto": ResolutiveRegimeRouter,
        "v5": ResolutiveV5,
        "v6": ResolutiveV6,
        "multires": ResolutiveHybridMultiResolution,
        "robust": ResolutiveHybridMultiResolutionRobust,
    }
    try:
        engine = engines[mode]()
    except KeyError as exc:
        raise ValueError(f"unknown optimization mode: {mode!r}") from exc

    return engine.minimize(
        objective,
        dimension=dimension,
        bounds=bounds,
        budget=budget,
        seed=seed,
    )
