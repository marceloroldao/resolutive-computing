import numpy as np

from resolutive.optimization.v2_spiral import ResolutiveV2Spiral
from resolutive.optimization.v4_historical import ResolutiveV4Historical
from resolutive.optimization.v5_spiral_conditional import ResolutiveV5ConditionalSpiral


def sphere(x):
    x = np.asarray(x)
    return float(np.dot(x, x))


def _check(optimizer):
    a = optimizer.minimize(sphere, dimension=4, bounds=(-5.0, 5.0), budget=700, seed=9)
    b = optimizer.minimize(sphere, dimension=4, bounds=(-5.0, 5.0), budget=700, seed=9)
    assert a.evaluations <= 700
    assert b.evaluations <= 700
    assert a.fun == b.fun
    assert np.allclose(a.x, b.x)
    assert np.isfinite(a.fun)


def test_v2_spiral_contract():
    _check(ResolutiveV2Spiral())


def test_v4_historical_contract():
    _check(ResolutiveV4Historical())


def test_v5_conditional_spiral_contract():
    _check(ResolutiveV5ConditionalSpiral())
