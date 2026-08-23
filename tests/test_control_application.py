import numpy as np

from resolutive.applications.control import decode_pid, pid_step_response_cost


def test_decode_pid_maps_normalized_vector_to_expected_ranges():
    assert decode_pid(np.array([-1.0, -1.0, -1.0])) == (0.0, 0.0, 0.0)
    assert decode_pid(np.array([1.0, 1.0, 1.0])) == (12.0, 6.0, 3.0)


def test_pid_cost_is_deterministic_and_finite():
    x = np.array([0.0, -0.2, -0.4])
    first = pid_step_response_cost(x)
    second = pid_step_response_cost(x)
    assert first == second
    assert np.isfinite(first)
    assert first > 0.0


def test_nonzero_pid_can_improve_over_zero_controller():
    zero = pid_step_response_cost(np.array([-1.0, -1.0, -1.0]))
    candidate = pid_step_response_cost(np.array([0.0, -0.2, -0.4]))
    assert candidate < zero
