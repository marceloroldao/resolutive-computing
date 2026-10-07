import numpy as np

from resolutive.optimization.hybrid_multires_adaptive import (
    ResolutiveHybridMultiResolutionAdaptive,
)


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


class AlternatingNoise:
    def __init__(self):
        self.k = 0

    def __call__(self, x: np.ndarray) -> float:
        base = float(np.sum(x * x))
        self.k += 1
        return base + (0.05 if self.k % 2 else -0.05)


def test_adaptive_is_deterministic_on_clean_objective_and_respects_budget():
    opt = ResolutiveHybridMultiResolutionAdaptive()
    a = opt.minimize(sphere, dimension=4, bounds=(-5.0, 5.0), budget=1000, seed=11)
    b = opt.minimize(sphere, dimension=4, bounds=(-5.0, 5.0), budget=1000, seed=11)
    assert a.evaluations <= 1000
    assert b.evaluations <= 1000
    assert np.allclose(a.x, b.x)
    assert a.fun == b.fun
    assert a.version == "RO-Hybrid-Multires-Adaptive-exp"


def test_adaptive_runs_with_repeatable_noise_and_respects_budget():
    opt = ResolutiveHybridMultiResolutionAdaptive(noise_ratio_threshold=0.01)
    result = opt.minimize(
        AlternatingNoise(), dimension=4, bounds=(-5.0, 5.0), budget=1100, seed=3
    )
    assert np.isfinite(result.fun)
    assert result.evaluations <= 1100


def test_adaptive_threshold_must_be_positive():
    try:
        ResolutiveHybridMultiResolutionAdaptive(noise_ratio_threshold=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero noise threshold should fail")
