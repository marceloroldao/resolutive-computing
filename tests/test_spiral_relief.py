import numpy as np
import pytest

from resolutive.benchmarks.functions import ackley
from resolutive.optimization.spiral_relief import SpiralReliefOptimizer, _fit_relief


def test_quadratic_relief_recovers_gradient_and_hessian():
    coords = np.array([
        [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
        [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
        [1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
    ])
    a, b = coords[:, 0], coords[:, 1]
    values = 3.0 + 2.0 * a - 4.0 * b + 5.0 * a * a + 1.5 * a * b + 2.0 * b * b
    gradient, hessian = _fit_relief(coords, values)
    assert np.allclose(gradient, [2.0, -4.0], atol=1e-10)
    assert np.allclose(hessian, [[10.0, 1.5], [1.5, 4.0]], atol=1e-10)


def test_spiral_relief_is_reproducible_and_respects_budget():
    opt = SpiralReliefOptimizer(spiral_points=12)
    first = opt.minimize(ackley, dimension=4, bounds=(-32.768, 32.768), budget=500, seed=7)
    second = opt.minimize(ackley, dimension=4, bounds=(-32.768, 32.768), budget=500, seed=7)
    assert first.evaluations <= 500
    assert first.evaluations == second.evaluations
    assert first.fun == second.fun
    assert np.array_equal(first.x, second.x)


def test_spiral_relief_improves_ackley_from_random_start():
    result = SpiralReliefOptimizer(spiral_points=12).minimize(
        ackley, dimension=4, bounds=(-32.768, 32.768), budget=700, seed=3
    )
    assert result.fun < 20.0


def test_spiral_relief_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        SpiralReliefOptimizer(spiral_points=6)
    with pytest.raises(ValueError):
        SpiralReliefOptimizer(turns=0.0)
    with pytest.raises(ValueError):
        SpiralReliefOptimizer().minimize(
            ackley, dimension=1, bounds=(-1.0, 1.0), budget=100, seed=0
        )
