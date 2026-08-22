import numpy as np
import pytest

from resolutive import OptimizationSession


def sphere_batch(points: np.ndarray) -> np.ndarray:
    return np.sum(points * points, axis=1)


def run_session(seed: int = 7):
    session = OptimizationSession(
        dimension=3,
        bounds=(-5.0, 5.0),
        budget=24,
        seed=seed,
        batch_size=6,
    )
    while not session.done:
        batch = session.ask()
        session.tell(sphere_batch(batch.points))
    return session.result()


def test_ask_tell_respects_budget_and_returns_result() -> None:
    result = run_session()
    assert result.evaluations == 24
    assert result.status == "success"
    assert result.optimizer == "RO-AskTell-Prototype"
    assert result.diagnostics is not None
    assert result.diagnostics["remaining_budget"] == 0
    assert np.isfinite(result.fun)


def test_ask_tell_is_deterministic() -> None:
    a = run_session(seed=19)
    b = run_session(seed=19)
    assert a.fun == pytest.approx(b.fun, rel=0.0, abs=0.0)
    assert np.array_equal(a.x, b.x)


def test_ask_requires_tell_before_next_batch() -> None:
    session = OptimizationSession(dimension=2, bounds=(-1.0, 1.0), budget=4, batch_size=2)
    session.ask()
    with pytest.raises(RuntimeError, match="tell"):
        session.ask()


def test_tell_validates_batch_size() -> None:
    session = OptimizationSession(dimension=2, bounds=(-1.0, 1.0), budget=4, batch_size=2)
    session.ask()
    with pytest.raises(ValueError, match="one scalar"):
        session.tell([1.0])
