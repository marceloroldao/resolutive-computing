import numpy as np
import pytest

from resolutive.benchmarks.functions import ackley
from resolutive.optimization.baselines import DifferentialEvolution, RandomSearch, SimulatedAnnealing
from resolutive.optimization.v2 import ResolutiveV2
from resolutive.optimization.v5 import ResolutiveV5
from resolutive.optimization.v6 import ResolutiveV6
from resolutive.optimization.v7 import ResolutiveV7


def test_random_search_uses_exact_budget():
    result = RandomSearch().minimize(
        ackley, dimension=4, bounds=(-32.768, 32.768), budget=123, seed=1
    )
    assert result.evaluations == 123
    assert result.fun >= 0.0


def test_simulated_annealing_uses_exact_budget():
    result = SimulatedAnnealing().minimize(
        ackley, dimension=4, bounds=(-32.768, 32.768), budget=123, seed=1
    )
    assert result.evaluations == 123
    assert result.fun >= 0.0


def test_differential_evolution_uses_exact_budget():
    result = DifferentialEvolution(population_multiplier=4).minimize(
        ackley, dimension=4, bounds=(-32.768, 32.768), budget=123, seed=1
    )
    assert result.evaluations == 123
    assert result.fun >= 0.0


def test_v2_respects_budget_and_improves_ackley():
    result = ResolutiveV2(population=16).minimize(
        ackley, dimension=4, bounds=(-32.768, 32.768), budget=500, seed=1
    )
    assert result.evaluations <= 500
    assert result.fun < 10.0


def test_v5_respects_budget_and_improves_ackley():
    result = ResolutiveV5(population=20).minimize(
        ackley, dimension=4, bounds=(-32.768, 32.768), budget=600, seed=1
    )
    assert result.evaluations <= 600
    assert result.fun < 10.0


def test_v6_respects_budget_and_improves_ackley():
    result = ResolutiveV6(population=20).minimize(
        ackley, dimension=4, bounds=(-32.768, 32.768), budget=600, seed=1
    )
    assert result.evaluations <= 600
    assert result.fun < 10.0


def test_v7_respects_budget_and_improves_ackley():
    result = ResolutiveV7(population=20).minimize(
        ackley, dimension=4, bounds=(-32.768, 32.768), budget=600, seed=1
    )
    assert result.evaluations <= 600
    assert result.fun < 10.0


def test_v5_is_reproducible_for_fixed_seed():
    optimizer = ResolutiveV5(population=20)
    first = optimizer.minimize(
        ackley, dimension=4, bounds=(-32.768, 32.768), budget=600, seed=7
    )
    second = optimizer.minimize(
        ackley, dimension=4, bounds=(-32.768, 32.768), budget=600, seed=7
    )
    assert first.evaluations == second.evaluations
    assert first.fun == second.fun
    assert np.array_equal(first.x, second.x)


def test_v6_is_reproducible_for_fixed_seed():
    optimizer = ResolutiveV6(population=20)
    first = optimizer.minimize(
        ackley, dimension=4, bounds=(-32.768, 32.768), budget=600, seed=7
    )
    second = optimizer.minimize(
        ackley, dimension=4, bounds=(-32.768, 32.768), budget=600, seed=7
    )
    assert first.evaluations == second.evaluations
    assert first.fun == second.fun
    assert np.array_equal(first.x, second.x)


def test_v7_is_reproducible_for_fixed_seed():
    optimizer = ResolutiveV7(population=20)
    first = optimizer.minimize(
        ackley, dimension=4, bounds=(-32.768, 32.768), budget=600, seed=7
    )
    second = optimizer.minimize(
        ackley, dimension=4, bounds=(-32.768, 32.768), budget=600, seed=7
    )
    assert first.evaluations == second.evaluations
    assert first.fun == second.fun
    assert np.array_equal(first.x, second.x)


def test_v5_ablation_switches_preserve_budget_contract():
    variants = [
        {"use_memory": False},
        {"use_golden_rotation": False},
        {"use_escape": False},
        {"use_reanchor": False},
        {"use_collapse": False},
    ]
    for options in variants:
        result = ResolutiveV5(population=20, **options).minimize(
            ackley, dimension=4, bounds=(-32.768, 32.768), budget=400, seed=2
        )
        assert result.evaluations <= 400
        assert result.fun >= 0.0


def test_optimizers_reject_invalid_scalar_bounds():
    optimizers = (
        RandomSearch(),
        SimulatedAnnealing(),
        DifferentialEvolution(),
        ResolutiveV2(),
        ResolutiveV5(),
        ResolutiveV6(),
        ResolutiveV7(),
    )
    for optimizer in optimizers:
        with pytest.raises(ValueError):
            optimizer.minimize(ackley, dimension=4, bounds=(1.0, 1.0), budget=100, seed=0)


def test_resolutive_optimizers_reject_too_small_budget():
    with pytest.raises(ValueError):
        ResolutiveV2(population=16).minimize(
            ackley, dimension=4, bounds=(-1.0, 1.0), budget=16, seed=0
        )
    with pytest.raises(ValueError):
        ResolutiveV5(population=20).minimize(
            ackley, dimension=4, bounds=(-1.0, 1.0), budget=20, seed=0
        )
    with pytest.raises(ValueError):
        ResolutiveV6(population=20).minimize(
            ackley, dimension=4, bounds=(-1.0, 1.0), budget=20, seed=0
        )
    with pytest.raises(ValueError):
        ResolutiveV7(population=20).minimize(
            ackley, dimension=4, bounds=(-1.0, 1.0), budget=20, seed=0
        )


def test_differential_evolution_rejects_budget_below_minimum_population():
    with pytest.raises(ValueError):
        DifferentialEvolution().minimize(
            ackley, dimension=4, bounds=(-1.0, 1.0), budget=3, seed=0
        )
