import numpy as np

from resolutive.benchmarks.functions import ackley, rastrigin, rosenbrock


def test_known_global_minima_are_zero():
    zero = np.zeros(10)
    one = np.ones(10)
    assert abs(ackley(zero)) < 1e-12
    assert abs(rastrigin(zero)) < 1e-12
    assert abs(rosenbrock(one)) < 1e-12
