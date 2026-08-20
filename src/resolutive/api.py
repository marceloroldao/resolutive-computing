"""Public Python API for Resolutive Computing.

This module is the compatibility boundary intended to converge toward v1.0.
Internal optimizer engines may evolve without requiring users to import their
implementation modules directly.
"""
from __future__ import annotations

from typing import Literal

from .hybrid_multires_session import HybridMultiResolutionSession
from .hybrid_session import HybridRegimeSession
from .optimization.common import Objective, OptimizationResult
from .optimization.hybrid_multires import ResolutiveHybridMultiResolution
from .optimization.hybrid_multires_robust import ResolutiveHybridMultiResolutionRobust
from .optimization.regime_router import ResolutiveRegimeRouter
from .optimization.v5 import ResolutiveV5
from .optimization.v6 import ResolutiveV6
from .session import OptimizationSession

Mode = Literal["auto", "v5", "v6", "multires", "robust"]
SessionMode = Literal["prototype", "multires", "hybrid"]


def optimize(
    objective: Objective,
    *,
    dimension: int,
    bounds: tuple[float, float],
    budget: int = 6000,
    seed: int = 0,
    mode: Mode = "auto",
) -> OptimizationResult:
    """Minimize a continuous black-box objective under an evaluation budget."""
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


def create_session(
    *,
    dimension: int,
    bounds: tuple[float, float],
    budget: int,
    seed: int = 0,
    mode: SessionMode = "multires",
    batch_size: int = 16,
):
    """Create an experimental stateful ask/tell optimization session.

    ``mode='multires'`` uses the Hybrid-backed multiresolution session.  The
    historical random-exploration multiresolution prototype remains in the
    repository for ablation, but is no longer the public default.
    """
    if mode == "hybrid":
        return HybridRegimeSession(
            dimension=dimension,
            bounds=bounds,
            budget=budget,
            seed=seed,
        )
    if mode == "multires":
        return HybridMultiResolutionSession(
            dimension=dimension,
            bounds=bounds,
            budget=budget,
            seed=seed,
        )
    if mode == "prototype":
        return OptimizationSession(
            dimension=dimension,
            bounds=bounds,
            budget=budget,
            seed=seed,
            batch_size=batch_size,
        )
    raise ValueError(f"unknown session mode: {mode!r}")
