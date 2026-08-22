import numpy as np
import pytest

from resolutive import checkpoint_json, create_session, restore_json
from resolutive.benchmarks.functions import ackley


def _evaluate(batch):
    return np.asarray([ackley(x) for x in batch.points], dtype=float)


def _finish(session):
    while not session.done:
        batch = session.ask()
        session.tell(_evaluate(batch))
    return session.result()


def test_hybrid_checkpoint_after_recon_preserves_exact_trajectory() -> None:
    kwargs = dict(
        dimension=4,
        bounds=(-32.768, 32.768),
        budget=1000,
        seed=31,
        mode="hybrid",
    )
    reference = create_session(**kwargs)
    resumed_source = create_session(**kwargs)

    for session in (reference, resumed_source):
        batch = session.ask()
        session.tell(_evaluate(batch))

    restored = restore_json(checkpoint_json(resumed_source))

    a = _finish(reference)
    b = _finish(restored)

    assert b.evaluations == a.evaluations
    assert b.fun == a.fun
    assert np.array_equal(b.x, a.x)
    assert b.diagnostics["core"] == a.diagnostics["core"]


def test_checkpoint_with_pending_ask_requires_tell_and_resumes_exactly() -> None:
    kwargs = dict(
        dimension=4,
        bounds=(-32.768, 32.768),
        budget=1200,
        seed=37,
        mode="multires",
    )
    reference = create_session(**kwargs)
    source = create_session(**kwargs)

    # Advance both sessions identically into the state machine.
    for _ in range(3):
        for session in (reference, source):
            batch = session.ask()
            session.tell(_evaluate(batch))

    ref_batch = reference.ask()
    src_batch = source.ask()
    assert np.array_equal(ref_batch.points, src_batch.points)

    restored = restore_json(checkpoint_json(source))

    # A pending batch must survive restart. Emitting new work before tell()
    # would duplicate a physical experiment.
    with pytest.raises(RuntimeError, match="tell"):
        restored.ask()

    values = _evaluate(src_batch)
    reference.tell(values)
    restored.tell(values)

    a = _finish(reference)
    b = _finish(restored)

    assert b.evaluations == a.evaluations
    assert b.fun == a.fun
    assert np.array_equal(b.x, a.x)
