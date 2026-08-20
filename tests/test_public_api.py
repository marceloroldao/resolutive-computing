import numpy as np
import pytest

from resolutive import OptimizationResult, create_session, optimize
from resolutive.multires_session import MultiResolutionSession
from resolutive.session import OptimizationSession


def sphere(x: np.ndarray) -> float:
    return float(np.dot(x, x))


@pytest.mark.parametrize("mode,budget", [
    ("v5", 1200),
    ("v6", 1200),
    ("multires", 1200),
    ("robust", 1400),
])
def test_public_api_specialists(mode: str, budget: int) -> None:
    result = optimize(
        sphere,
        dimension=4,
        bounds=(-5.0, 5.0),
        budget=budget,
        seed=17,
        mode=mode,
    )
    assert isinstance(result, OptimizationResult)
    assert result.x.shape == (4,)
    assert np.isfinite(result.fun)
    assert result.evaluations <= budget


def test_public_api_is_deterministic_for_specialist() -> None:
    kwargs = dict(dimension=4, bounds=(-5.0, 5.0), budget=1200, seed=23, mode="multires")
    a = optimize(sphere, **kwargs)
    b = optimize(sphere, **kwargs)
    assert a.fun == pytest.approx(b.fun, rel=0.0, abs=0.0)
    assert np.array_equal(a.x, b.x)
    assert a.evaluations == b.evaluations


def test_public_api_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown optimization mode"):
        optimize(sphere, dimension=4, bounds=(-5.0, 5.0), budget=1200, mode="invalid")  # type: ignore[arg-type]


def test_public_session_factory_defaults_to_multires() -> None:
    session = create_session(dimension=3, bounds=(-2.0, 2.0), budget=200)
    assert isinstance(session, MultiResolutionSession)


def test_public_session_factory_preserves_prototype() -> None:
    session = create_session(
        dimension=3,
        bounds=(-2.0, 2.0),
        budget=50,
        mode="prototype",
    )
    assert isinstance(session, OptimizationSession)


def test_public_session_factory_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown session mode"):
        create_session(dimension=3, bounds=(-2.0, 2.0), budget=200, mode="invalid")  # type: ignore[arg-type]
