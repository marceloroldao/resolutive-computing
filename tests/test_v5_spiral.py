import numpy as np

from resolutive.optimization.v5_spiral import ResolutiveV5Spiral


def sphere(x: np.ndarray) -> float:
    return float(np.dot(x, x))


def test_v5_spiral_is_reproducible():
    opt = ResolutiveV5Spiral(spiral_fraction=0.15)
    a = opt.minimize(sphere, dimension=4, bounds=(-5.0, 5.0), budget=800, seed=7)
    b = opt.minimize(sphere, dimension=4, bounds=(-5.0, 5.0), budget=800, seed=7)
    assert a.fun == b.fun
    assert np.allclose(a.x, b.x)


def test_v5_spiral_respects_budget():
    result = ResolutiveV5Spiral().minimize(
        sphere, dimension=4, bounds=(-5.0, 5.0), budget=800, seed=3
    )
    assert result.evaluations <= 800
    assert result.version == "RO-V5-Spiral-exp"
