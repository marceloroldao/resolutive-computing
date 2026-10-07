import numpy as np

from resolutive.benchmarks.functions import ackley, rastrigin
from resolutive.benchmarks.transforms import GaussianNoisyObjective, orthogonal_matrix, shifted_rotated


def test_orthogonal_matrix_is_orthogonal_and_proper():
    q = orthogonal_matrix(6, seed=11)
    assert np.allclose(q.T @ q, np.eye(6), atol=1e-12)
    assert np.isclose(np.linalg.det(q), 1.0, atol=1e-12)


def test_shifted_rotated_ackley_preserves_known_minimum_value():
    transformed, shift, _ = shifted_rotated(
        ackley,
        dimension=5,
        bounds=(-32.768, 32.768),
        seed=123,
    )
    assert np.isclose(transformed(shift), ackley(np.zeros(5)), atol=1e-12)


def test_shifted_rotated_rastrigin_is_deterministic_for_instance_seed():
    first, shift1, rotation1 = shifted_rotated(
        rastrigin, dimension=4, bounds=(-5.12, 5.12), seed=9
    )
    second, shift2, rotation2 = shifted_rotated(
        rastrigin, dimension=4, bounds=(-5.12, 5.12), seed=9
    )
    x = np.array([0.1, -0.2, 0.3, -0.4])
    assert np.array_equal(shift1, shift2)
    assert np.array_equal(rotation1, rotation2)
    assert first(x) == second(x)


def test_gaussian_noisy_objective_replays_same_noise_sequence():
    x1 = np.array([0.2, -0.1, 0.4])
    x2 = np.array([0.0, 0.1, -0.2])
    a = GaussianNoisyObjective(ackley, sigma=0.5, seed=77)
    b = GaussianNoisyObjective(ackley, sigma=0.5, seed=77)
    assert a(x1) == b(x1)
    assert a(x2) == b(x2)
