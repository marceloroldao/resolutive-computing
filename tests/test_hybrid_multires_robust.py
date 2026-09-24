import numpy as np

from resolutive.optimization.hybrid_multires_robust import (
    ResolutiveHybridMultiResolutionRobust,
    _mad_clip,
)


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def test_robust_multires_is_deterministic_and_respects_budget():
    opt = ResolutiveHybridMultiResolutionRobust()
    a = opt.minimize(sphere, dimension=4, bounds=(-5.0, 5.0), budget=1200, seed=11)
    b = opt.minimize(sphere, dimension=4, bounds=(-5.0, 5.0), budget=1200, seed=11)
    assert a.evaluations <= 1200
    assert b.evaluations <= 1200
    assert np.allclose(a.x, b.x)
    assert a.fun == b.fun
    assert a.version == "RO-Hybrid-Multires-Robust-exp"


def test_mad_clip_limits_impulsive_outlier():
    values = np.array([1.00, 1.02, 0.98, 1.01, 25.0])
    clipped = _mad_clip(values, 3.0)
    assert clipped[-1] < 2.0
    assert np.allclose(clipped[:4], values[:4])


def test_robust_multires_rejects_too_small_budget():
    try:
        ResolutiveHybridMultiResolutionRobust().minimize(
            sphere, dimension=4, bounds=(-5.0, 5.0), budget=900, seed=0
        )
    except ValueError:
        pass
    else:
        raise AssertionError("budget below 1000 should fail")
