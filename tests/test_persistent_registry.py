import numpy as np
import pytest

from resolutive.persistent_registry import PersistentSessionRegistry


def _sphere(points: np.ndarray) -> list[float]:
    return np.sum(points * points, axis=1).tolist()


def test_persistent_registry_survives_restart_with_pending_ask(tmp_path) -> None:
    first = PersistentSessionRegistry(tmp_path)
    sid = first.create(
        session_id="alpha",
        dimension=3,
        bounds=(-5.0, 5.0),
        budget=8,
        seed=7,
        mode="prototype",
        batch_size=4,
    )

    batch = first.ask(sid)
    points = batch.points.copy()

    restarted = PersistentSessionRegistry(tmp_path)
    assert restarted.ids() == ("alpha",)
    with pytest.raises(RuntimeError, match="tell"):
        restarted.ask(sid)

    restarted.tell(sid, _sphere(points))
    info = restarted.info(sid)
    assert info.evaluations == 4
    assert info.remaining == 4

    again = PersistentSessionRegistry(tmp_path)
    next_batch = again.ask(sid)
    again.tell(sid, _sphere(next_batch.points))
    result = again.result(sid)
    assert result.evaluations == 8
    assert result.status == "success"


def test_persistent_registry_restores_multiple_sessions_and_delete(tmp_path) -> None:
    registry = PersistentSessionRegistry(tmp_path)
    a = registry.create(
        session_id="a",
        dimension=2,
        bounds=(-5.0, 5.0),
        budget=4,
        seed=1,
        mode="prototype",
        batch_size=2,
    )
    b = registry.create(
        session_id="b",
        dimension=2,
        bounds=(-5.0, 5.0),
        budget=4,
        seed=2,
        mode="prototype",
        batch_size=2,
    )

    for sid in (a, b):
        batch = registry.ask(sid)
        registry.tell(sid, _sphere(batch.points))

    restarted = PersistentSessionRegistry(tmp_path)
    assert set(restarted.ids()) == {"a", "b"}
    assert restarted.info("a").evaluations == 2
    assert restarted.info("b").evaluations == 2

    restarted.delete("a")
    after_delete = PersistentSessionRegistry(tmp_path)
    assert after_delete.ids() == ("b",)
    with pytest.raises(KeyError, match="unknown session id"):
        after_delete.get("a")
