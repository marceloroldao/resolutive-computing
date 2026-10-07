"""Regression guard for the planned Resolutive Computing v1 public surface."""

from __future__ import annotations

import resolutive


V1_PUBLIC_NAMES = {
    "optimize",
    "create_session",
    "checkpoint_session",
    "restore_session",
    "checkpoint_json",
    "restore_json",
    "SessionRegistry",
    "PersistentSessionRegistry",
    "OptimizationResult",
    "OptimizationSession",
    "MultiResolutionSession",
    "MultiResolutionState",
    "AskBatch",
    "ResolutiveV2",
    "ResolutiveV5",
}


def test_planned_v1_public_names_are_exported() -> None:
    exported = set(resolutive.__all__)
    assert V1_PUBLIC_NAMES <= exported
    for name in V1_PUBLIC_NAMES:
        assert hasattr(resolutive, name), name
