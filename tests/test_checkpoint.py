import json

import numpy as np

from resolutive import checkpoint_json, create_session, restore_json
from resolutive.benchmarks.functions import ackley


def _step(session, objective, steps: int) -> None:
    for _ in range(steps):
        if session.done:
            return
        batch = session.ask()
        values = np.asarray([objective(x) for x in batch.points], dtype=float)
        session.tell(values)


def _finish(session, objective):
    while not session.done:
        batch = session.ask()
        values = np.asarray([objective(x) for x in batch.points], dtype=float)
        session.tell(values)
    return session.result()


def test_multires_checkpoint_roundtrip_preserves_exact_trajectory() -> None:
    kwargs = dict(
        dimension=4,
        bounds=(-32.768, 32.768),
        budget=1200,
        seed=19,
        mode="multires",
    )
    uninterrupted = create_session(**kwargs)
    resumed_source = create_session(**kwargs)

    _step(uninterrupted, ackley, 8)
    _step(resumed_source, ackley, 8)

    payload = checkpoint_json(resumed_source)
    parsed = json.loads(payload)
    assert parsed["format"] == "resolutive-session"
    assert parsed["version"] == 1

    restored = restore_json(payload)

    a = _finish(uninterrupted, ackley)
    b = _finish(restored, ackley)

    assert b.evaluations == a.evaluations
    assert b.fun == a.fun
    assert np.array_equal(b.x, a.x)


def test_checkpoint_rejects_unknown_version() -> None:
    session = create_session(
        dimension=4,
        bounds=(-5.0, 5.0),
        budget=1000,
        seed=3,
        mode="hybrid",
    )
    payload = json.loads(checkpoint_json(session))
    payload["version"] = 999

    try:
        restore_json(json.dumps(payload))
    except ValueError as exc:
        assert "version" in str(exc)
    else:
        raise AssertionError("unknown checkpoint version must be rejected")
