import numpy as np

from resolutive.benchmarks.functions import ackley
from resolutive.hybrid_multires_sequential_session import HybridMultiResolutionSequentialSession
from resolutive.hybrid_sequential_session import HybridRegimeSequentialSession
from resolutive.optimization.hybrid_multires import ResolutiveHybridMultiResolution
from resolutive.optimization.hybrid_regime import ResolutiveHybridRegime
from resolutive.optimization.v2 import ResolutiveV2
from resolutive.v2_sequential_session import V2SequentialSession


def _run_session(session, objective):
    while not session.done:
        batch = session.ask()
        values = np.asarray([objective(x) for x in batch.points], dtype=float)
        session.tell(values)
    return session.result()


def _assert_exact_equivalence(monolithic, incremental) -> None:
    assert incremental.evaluations == monolithic.evaluations
    assert incremental.fun == monolithic.fun
    assert np.array_equal(incremental.x, monolithic.x)


def test_v2_ask_tell_is_exactly_equivalent_to_monolithic() -> None:
    dimension = 4
    bounds = (-32.768, 32.768)
    budget = 1200
    seed = 7
    mono = ResolutiveV2().minimize(
        ackley, dimension=dimension, bounds=bounds, budget=budget, seed=seed
    )
    inc = _run_session(
        V2SequentialSession(
            dimension=dimension, bounds=bounds, budget=budget, seed=seed
        ),
        ackley,
    )
    _assert_exact_equivalence(mono, inc)


def test_hybrid_ask_tell_is_exactly_equivalent_to_monolithic() -> None:
    dimension = 4
    bounds = (-32.768, 32.768)
    budget = 1200
    seed = 7
    mono = ResolutiveHybridRegime().minimize(
        ackley, dimension=dimension, bounds=bounds, budget=budget, seed=seed
    )
    inc = _run_session(
        HybridRegimeSequentialSession(
            dimension=dimension, bounds=bounds, budget=budget, seed=seed
        ),
        ackley,
    )
    _assert_exact_equivalence(mono, inc)


def test_multires_ask_tell_is_exactly_equivalent_to_monolithic() -> None:
    dimension = 4
    bounds = (-32.768, 32.768)
    budget = 1200
    seed = 7
    mono = ResolutiveHybridMultiResolution().minimize(
        ackley, dimension=dimension, bounds=bounds, budget=budget, seed=seed
    )
    inc = _run_session(
        HybridMultiResolutionSequentialSession(
            dimension=dimension, bounds=bounds, budget=budget, seed=seed
        ),
        ackley,
    )
    _assert_exact_equivalence(mono, inc)
