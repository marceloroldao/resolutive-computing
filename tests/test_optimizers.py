from resolutive.benchmarks.functions import ackley
from resolutive.optimization.v2 import ResolutiveV2
from resolutive.optimization.v5 import ResolutiveV5


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
