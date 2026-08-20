import numpy as np

from resolutive import create_session


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
