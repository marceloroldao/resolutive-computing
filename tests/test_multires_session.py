import numpy as np
import pytest

from resolutive.multires_session import MultiResolutionSession


def sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x * x, axis=1)


def run_session(seed: int = 7):
    session = MultiResolutionSession(
        dimension=4,
        bounds=(-5.0, 5.0),
        budget=240,
        seed=seed,
        batch_size=12,
    )
    phases = []
    while not session.done:
        batch = session.ask()
        phases.append(session.state.phase)
        session.tell(sphere(batch.points))
    return session, phases


def test_multires_session_respects_budget_and_progresses() -> None:
    session, phases = run_session()
    result = session.result()
    assert result.evaluations == 240
    assert result.status == "success"
    assert result.optimizer == "RO-Multires-AskTell-exp"
    assert np.isfinite(result.fun)
    assert "explore" in phases
    assert any(p in {"spiral", "slide", "polish"} for p in phases)


def test_multires_session_is_deterministic() -> None:
    a, _ = run_session(seed=19)
    b, _ = run_session(seed=19)
    ra, rb = a.result(), b.result()
    assert ra.fun == pytest.approx(rb.fun, rel=0.0, abs=0.0)
    assert np.array_equal(ra.x, rb.x)
    assert ra.evaluations == rb.evaluations


def test_multires_session_requires_tell_before_next_ask() -> None:
    session = MultiResolutionSession(dimension=3, bounds=(-2.0, 2.0), budget=200)
    session.ask()
    with pytest.raises(RuntimeError, match="tell"):
        session.ask()


def test_multires_session_rejects_wrong_value_count() -> None:
    session = MultiResolutionSession(dimension=3, bounds=(-2.0, 2.0), budget=200)
    batch = session.ask()
    with pytest.raises(ValueError, match="one scalar"):
        session.tell(np.zeros(len(batch.points) + 1))
