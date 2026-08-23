import numpy as np

from resolutive.optimization.hybrid_regime import ResolutiveHybridRegime, _reconnaissance


def sphere(x):
    x = np.asarray(x)
    return float(np.dot(x, x))


def rugged(x):
    x = np.asarray(x)
    return float(np.sum(x * x - 4.0 * np.cos(7.0 * x)))


def test_reconnaissance_is_reproducible():
    a = _reconnaissance(sphere, dimension=5, bounds=(-5.0, 5.0), budget=50, seed=7)
    b = _reconnaissance(sphere, dimension=5, bounds=(-5.0, 5.0), budget=50, seed=7)
    assert a[0] == b[0]
    assert np.allclose(a[1], b[1])
    assert a[2] == b[2]
    assert a[3] == b[3]


def test_hybrid_respects_budget_and_seed():
    opt = ResolutiveHybridRegime()
    a = opt.minimize(sphere, dimension=6, bounds=(-5.0, 5.0), budget=900, seed=11)
    b = opt.minimize(sphere, dimension=6, bounds=(-5.0, 5.0), budget=900, seed=11)
    assert a.evaluations <= 900
    assert b.evaluations <= 900
    assert a.fun == b.fun
    assert np.allclose(a.x, b.x)
    assert a.version.startswith("RO-Hybrid-exp[")


def test_ruggedness_measure_is_bounded():
    score, *_ = _reconnaissance(rugged, dimension=4, bounds=(-5.0, 5.0), budget=49, seed=3)
    assert 0.0 <= score <= 1.0
