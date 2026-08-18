from resolutive.benchmarks.functions import ackley
from resolutive.optimization.baselines import RandomSearch, SimulatedAnnealing
from resolutive.optimization.v2 import ResolutiveV2
from resolutive.optimization.v5 import ResolutiveV5


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
