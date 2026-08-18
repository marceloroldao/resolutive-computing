"""Strong external-baseline campaign for Resolutive Optimizers.

Uses maintained third-party implementations rather than the repository's
pedagogical baselines: pycma CMA-ES and scipy.optimize.differential_evolution.
All methods receive the same objective-evaluation budget.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cma
import numpy as np
from scipy.optimize import differential_evolution

from resolutive.benchmarks.functions import DEFAULT_BENCHMARKS
from resolutive.optimization.v2 import ResolutiveV2
from resolutive.optimization.v5 import ResolutiveV5
from resolutive.optimization.v6 import ResolutiveV6


def _cma_es(objective, dimension, bounds, budget, seed):
    lo, hi = bounds
    x0 = np.full(dimension, (lo + hi) / 2.0)
    sigma0 = (hi - lo) / 4.0
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        "bounds": [lo, hi], "seed": seed + 1, "verbose": -9,
        "verb_disp": 0, "maxfevals": budget,
    })
    used = 0
    best = float("inf")
    while not es.stop() and used < budget:
        xs = es.ask()
        xs = xs[: max(0, budget - used)]
        if not xs:
            break
        ys = [float(objective(np.asarray(x))) for x in xs]
        used += len(ys)
        best = min(best, min(ys))
        es.tell(xs, ys)
    return best, used


def _scipy_de(objective, dimension, bounds, budget, seed):
    popsize = 10
    pop_n = popsize * dimension
    maxiter = max(0, budget // pop_n - 1)
    res = differential_evolution(
        objective, [(bounds[0], bounds[1])] * dimension,
        seed=seed, popsize=popsize, maxiter=maxiter, polish=False,
        updating="immediate", workers=1, tol=0.0, atol=0.0,
    )
    return float(res.fun), int(res.nfev)


def run(dimension: int, budget: int, seeds: int, output: Path) -> None:
    methods = {"CMA-ES(pycma)": _cma_es, "DE(scipy)": _scipy_de}
    ro = {"RO-V2": ResolutiveV2(), "RO-V5": ResolutiveV5(), "RO-V6": ResolutiveV6()}
    rows = []
    for benchmark_name, (objective, bounds) in DEFAULT_BENCHMARKS.items():
        for name in [*methods, *ro]:
            vals, evals = [], []
            for seed in range(seeds):
                if name in methods:
                    val, used = methods[name](objective, dimension, bounds, budget, seed)
                else:
                    result = ro[name].minimize(objective, dimension=dimension, bounds=bounds, budget=budget, seed=seed)
                    val, used = float(result.fun), int(result.evaluations)
                vals.append(val); evals.append(used)
            rows.append({
                "benchmark": benchmark_name, "optimizer": name,
                "dimension": dimension, "budget": budget, "seeds": seeds,
                "median": float(np.median(vals)), "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if seeds > 1 else 0.0,
                "best": float(np.min(vals)), "worst": float(np.max(vals)),
                "max_evaluations_used": int(np.max(evals)),
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dimension", type=int, default=10)
    p.add_argument("--budget", type=int, default=3000)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--output", type=Path, default=Path("results/strong-baselines.csv"))
    a = p.parse_args(); run(a.dimension, a.budget, a.seeds, a.output)
