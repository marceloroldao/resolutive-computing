import numpy as np

from resolutive.optimization.hybrid_multires import ResolutiveHybridMultiResolution


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def test_multires_is_deterministic_and_respects_budget():
    opt = ResolutiveHybridMultiResolution()
    a = opt.minimize(sphere, dimension=4, bounds=(-5.0, 5.0), budget=900, seed=7)
    b = opt.minimize(sphere, dimension=4, bounds=(-5.0, 5.0), budget=900, seed=7)
    assert a.evaluations <= 900
    assert b.evaluations <= 900
    assert np.allclose(a.x, b.x)
    assert a.fun == b.fun
    assert a.version == "RO-Hybrid-Multires-exp"


def test_radius_schedule_must_shrink():
    try:
        ResolutiveHybridMultiResolution(radius_schedule=(0.04, 0.05))
    except ValueError:
        pass
    else:
        raise AssertionError("non-shrinking schedule should fail")
