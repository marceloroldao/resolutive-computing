"""Hybrid MultiResolution ask/tell session with a sequentially faithful V2 core."""
from __future__ import annotations

from .hybrid_multires_session import HybridMultiResolutionSession
from .hybrid_sequential_session import HybridRegimeSequentialSession


class HybridMultiResolutionSequentialSession(HybridMultiResolutionSession):
    """Multires session whose hybrid core preserves V2 sequential collapse."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.core = HybridRegimeSequentialSession(
            dimension=self.dimension,
            bounds=(self.lo, self.hi),
            budget=self.core_budget,
            seed=self.seed,
            local_fraction=0.10,
        )
