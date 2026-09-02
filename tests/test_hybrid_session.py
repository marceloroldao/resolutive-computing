import numpy as np

from resolutive import create_session
from resolutive.hybrid_sequential_session import HybridRegimeSequentialSession


def sphere(x: np.ndarray) -> float:
    return float(np.dot(x, x))


def run_once(seed: int):
    session = create_session(
        dimension=4,
        bounds=(-5.0, 5.0),
        budget=1000,
        seed=seed,
        mode="hybrid",
    )
    while not session.done:
        batch = session.ask()
        values = np.asarray([sphere(x) for x in batch.points], dtype=float)
        session.tell(values)
    return session.result()


def test_hybrid_session_respects_budget_and_returns_finite_result() -> None:
    result = run_once(11)
    assert result.evaluations <= 1000
    assert np.isfinite(result.fun)
    assert result.status == "success"
    assert result.diagnostics["core"] in {"V2", "V5"}


def test_hybrid_session_is_deterministic() -> None:
    a = run_once(17)
    b = run_once(17)
    assert a.fun == b.fun
    assert np.array_equal(a.x, b.x)
    assert a.evaluations == b.evaluations
    assert a.diagnostics["core"] == b.diagnostics["core"]


def _forced_sequential_collapse(*, evaluations: int, best_fun: float) -> HybridRegimeSequentialSession:
    session = HybridRegimeSequentialSession(
        dimension=2,
        bounds=(-5.0, 5.0),
        budget=600,
        seed=3,
    )
    session.phase = "collapse"
    session.evaluations = evaluations
    session.best_x = np.zeros(2, dtype=float)
    session.best_fun = float(best_fun)
    session.collapse_step = 1.0
    session.pending = None
    session.pending_kind = None
    return session


def test_sequential_hybrid_collapse_updates_origin_immediately() -> None:
    session = _forced_sequential_collapse(evaluations=596, best_fun=10.0)

    first = session.ask()
    assert np.array_equal(first.points[0], np.array([-1.0, 0.0]))
    session.tell(np.array([5.0]))

    # The +axis-0 candidate must be built from the improved incumbent (-1, 0),
    # therefore it returns to (0, 0). A batch/parallel collapse would emit (1, 0).
    second = session.ask()
    assert np.array_equal(second.points[0], np.array([0.0, 0.0]))
    session.tell(np.array([4.0]))

    third = session.ask()
    assert np.array_equal(third.points[0], np.array([0.0, -1.0]))
    session.tell(np.array([6.0]))
    fourth = session.ask()
    assert np.array_equal(fourth.points[0], np.array([0.0, 1.0]))
    session.tell(np.array([6.0]))

    # At least one improvement occurred during the full sweep, so the monolithic
    # policy keeps the radius unchanged.
    assert session.collapse_step == 1.0
    assert session.done


def test_sequential_hybrid_collapse_shrinks_only_after_failed_full_sweep() -> None:
    session = _forced_sequential_collapse(evaluations=592, best_fun=0.0)

    for _ in range(4):
        batch = session.ask()
        assert len(batch.points) == 1
        session.tell(np.array([1.0]))

    assert session.collapse_step == 0.5
    assert not session.done
