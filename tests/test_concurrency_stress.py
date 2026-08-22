from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from resolutive.persistent_registry import PersistentSessionRegistry


def _sphere(points: np.ndarray) -> list[float]:
    return np.sum(points * points, axis=1).astype(float).tolist()


def _advance_once(registry: PersistentSessionRegistry, session_id: str) -> int:
    batch = registry.ask(session_id)
    registry.tell(session_id, _sphere(batch.points))
    return len(batch.points)


def test_many_independent_sessions_can_advance_concurrently(tmp_path: Path) -> None:
    registry = PersistentSessionRegistry(tmp_path)
    session_ids = [
        registry.create(
            session_id=f"s{i}",
            dimension=4,
            bounds=(-5.0, 5.0),
            budget=64,
            seed=100 + i,
            mode="prototype",
            batch_size=4,
        )
        for i in range(24)
    ]

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(_advance_once, registry, sid) for sid in session_ids for _ in range(4)]
        completed = [future.result() for future in futures]

    assert completed == [4] * (24 * 4)
    for sid in session_ids:
        info = registry.info(sid)
        assert info.evaluations == 16
        assert info.remaining == 48

    restarted = PersistentSessionRegistry(tmp_path)
    assert set(restarted.ids()) == set(session_ids)
    for sid in session_ids:
        info = restarted.info(sid)
        assert info.evaluations == 16
        assert info.remaining == 48


def test_concurrent_ask_on_same_session_allows_only_one_pending_batch(tmp_path: Path) -> None:
    registry = PersistentSessionRegistry(tmp_path)
    sid = registry.create(
        session_id="shared",
        dimension=3,
        bounds=(-5.0, 5.0),
        budget=16,
        seed=7,
        mode="prototype",
        batch_size=4,
    )

    def try_ask():
        try:
            return ("ok", registry.ask(sid))
        except RuntimeError as exc:
            return ("conflict", str(exc))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [future.result() for future in [pool.submit(try_ask) for _ in range(8)]]

    ok = [value for kind, value in results if kind == "ok"]
    conflicts = [value for kind, value in results if kind == "conflict"]
    assert len(ok) == 1
    assert len(conflicts) == 7
    assert all("tell() must be called" in message for message in conflicts)

    registry.tell(sid, _sphere(ok[0].points))
    assert registry.info(sid).evaluations == 4


def test_restart_preserves_pending_batch_under_concurrent_server_use(tmp_path: Path) -> None:
    registry = PersistentSessionRegistry(tmp_path)
    sid = registry.create(
        session_id="restart-pending",
        dimension=5,
        bounds=(-3.0, 3.0),
        budget=20,
        seed=12,
        mode="prototype",
        batch_size=5,
    )
    batch = registry.ask(sid)

    restarted = PersistentSessionRegistry(tmp_path)
    with pytest.raises(RuntimeError, match="tell\(\) must be called"):
        restarted.ask(sid)

    restarted.tell(sid, _sphere(batch.points))
    info = restarted.info(sid)
    assert info.evaluations == 5
    assert info.remaining == 15
