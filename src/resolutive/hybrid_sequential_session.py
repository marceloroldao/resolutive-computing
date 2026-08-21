"""Hybrid ask/tell session using the causally faithful sequential V2 port."""
from __future__ import annotations

from .hybrid_session import HybridRegimeSession
from .v2_sequential_session import V2SequentialSession


class HybridRegimeSequentialSession(HybridRegimeSession):
    """HybridRegimeSession with sequential V2 collapse when V2 is selected."""

    def tell(self, values) -> None:
        was_recon = self.phase == "recon"
        super().tell(values)
        if was_recon and self.core_name == "V2":
            self.core = V2SequentialSession(
                dimension=self.dimension,
                bounds=(self.lo, self.hi),
                budget=self.core_budget,
                seed=self.seed,
            )
