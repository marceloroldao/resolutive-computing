import numpy as np

from resolutive.optimization.regime_router import ResolutiveRegimeRouter


def sphere(x):
    x = np.asarray(x, dtype=float)
    return float(np.dot(x, x))


class AlternatingNoise:
    def __init__(self, amplitude=1.0):
        self.i = 0
        self.amplitude = amplitude

    def __call__(self, x):
        self.i += 1
        return sphere(x) + (self.amplitude if self.i % 2 else -self.amplitude)


def test_router_is_deterministic_on_clean_objective():
    a = ResolutiveRegimeRouter()
    b = ResolutiveRegimeRouter()
    ra = a.minimize(sphere, dimension=4, bounds=(-5.0, 5.0), budget=1400, seed=7)
    rb = b.minimize(sphere, dimension=4, bounds=(-5.0, 5.0), budget=1400, seed=7)
    assert np.allclose(ra.x, rb.x)
    assert ra.fun == rb.fun
    assert a.last_diagnostics is not None
    assert b.last_diagnostics is not None
    assert a.last_diagnostics.selected == b.last_diagnostics.selected


def test_router_respects_budget():
    router = ResolutiveRegimeRouter()
    result = router.minimize(sphere, dimension=4, bounds=(-5.0, 5.0), budget=1400, seed=3)
    assert result.evaluations <= 1400


def test_router_detects_more_noise_for_alternating_objective():
    clean = ResolutiveRegimeRouter()
    noisy = ResolutiveRegimeRouter()
    clean.minimize(sphere, dimension=4, bounds=(-5.0, 5.0), budget=1400, seed=11)
    noisy.minimize(AlternatingNoise(3.0), dimension=4, bounds=(-5.0, 5.0), budget=1400, seed=11)
    assert clean.last_diagnostics is not None
    assert noisy.last_diagnostics is not None
    assert noisy.last_diagnostics.noise_ratio > clean.last_diagnostics.noise_ratio
