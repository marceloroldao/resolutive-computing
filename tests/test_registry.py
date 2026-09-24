import numpy as np

from resolutive.benchmarks.functions import ackley, rastrigin
from resolutive.registry import SessionRegistry


def _evaluate(registry: SessionRegistry, session_id: str, objective) -> None:
    batch = registry.ask(session_id)
    values = np.asarray([objective(x) for x in batch.points], dtype=float)
    registry.tell(session_id, values)


def _finish(registry: SessionRegistry, session_id: str, objective):
    while not registry.info(session_id).done:
        _evaluate(registry, session_id, objective)
    return registry.result(session_id)


def test_registry_sessions_remain_isolated_when_interleaved() -> None:
    registry = SessionRegistry()
    a = registry.create(
        session_id="a",
        dimension=4,
        bounds=(-32.768, 32.768),
        budget=1200,
        seed=7,
        mode="multires",
    )
    b = registry.create(
        session_id="b",
        dimension=4,
        bounds=(-5.12, 5.12),
        budget=1200,
        seed=11,
        mode="multires",
    )

    for _ in range(5):
        _evaluate(registry, a, ackley)
        _evaluate(registry, b, rastrigin)

    a_checkpoint = registry.checkpoint(a)
    b_checkpoint = registry.checkpoint(b)

    reference_a = SessionRegistry()
    reference_b = SessionRegistry()
    ra = reference_a.restore(a_checkpoint, session_id="ra")
    rb = reference_b.restore(b_checkpoint, session_id="rb")

    result_a = _finish(registry, a, ackley)
    result_b = _finish(registry, b, rastrigin)
    ref_a = _finish(reference_a, ra, ackley)
    ref_b = _finish(reference_b, rb, rastrigin)

    assert result_a.evaluations == ref_a.evaluations
    assert result_a.fun == ref_a.fun
    assert np.array_equal(result_a.x, ref_a.x)

    assert result_b.evaluations == ref_b.evaluations
    assert result_b.fun == ref_b.fun
    assert np.array_equal(result_b.x, ref_b.x)


def test_registry_rejects_duplicate_and_unknown_ids() -> None:
    registry = SessionRegistry()
    registry.create(
        session_id="fixed",
        dimension=2,
        bounds=(-5.0, 5.0),
        budget=1000,
        seed=1,
        mode="hybrid",
    )

    try:
        registry.create(
            session_id="fixed",
            dimension=2,
            bounds=(-5.0, 5.0),
            budget=1000,
            seed=2,
            mode="hybrid",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate session ids must be rejected")

    try:
        registry.get("missing")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown session ids must be rejected")
