"""Release-candidate benchmark for Resolutive Computing v1.0.

The campaign compares the primary Resolutive engine against maintained,
widely-used derivative-free optimizers under a strict objective-call budget.
It intentionally excludes historical Resolutive variants from the ranking.

Default campaign:
- Ackley, Rastrigin, Rosenbrock (repository canonical implementations)
- dimensions 2, 10, 30
- budgets 1000, 3000, 9000 respectively (300 evaluations/dimension minimum,
  with a 1000-call floor)
- 10 seeds
- RO-Hybrid-Multires, CMA-ES (pycma), Differential Evolution (SciPy),
  Nelder-Mead (SciPy), Powell (SciPy)

Outputs a raw per-run CSV and an aggregated summary CSV. Lower objective values
are better. No claim of universal superiority should be made from this suite.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cma
import numpy as np
from scipy.optimize import differential_evolution, minimize

from resolutive.benchmarks.functions import DEFAULT_BENCHMARKS
from resolutive.optimization.hybrid_multires import ResolutiveHybridMultiResolution


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class BudgetedObjective:
    objective: Callable[[np.ndarray], float]
    budget: int
    used: int = 0
    best: float = float("inf")

    def __call__(self, x) -> float:
        if self.used >= self.budget:
            raise BudgetExhausted
        value = float(self.objective(np.asarray(x, dtype=float)))
        self.used += 1
        if value < self.best:
            self.best = value
        return value


def _seeded_start(dimension: int, bounds: tuple[float, float], seed: int) -> np.ndarray:
    lo, hi = bounds
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=dimension)


def _resolutive(objective, dimension, bounds, budget, seed):
    result = ResolutiveHybridMultiResolution().minimize(
        objective,
        dimension=dimension,
        bounds=bounds,
        budget=budget,
        seed=seed,
    )
    return float(result.fun), int(result.evaluations)


def _cma_es(objective, dimension, bounds, budget, seed):
    lo, hi = bounds
    wrapped = BudgetedObjective(objective, budget)
    x0 = _seeded_start(dimension, bounds, seed)
    sigma0 = (hi - lo) / 4.0
    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "bounds": [lo, hi],
            "seed": seed + 1,
            "verbose": -9,
            "verb_disp": 0,
            "maxfevals": budget,
        },
    )
    while not es.stop() and wrapped.used < budget:
        xs = es.ask()
        remaining = budget - wrapped.used
        xs = xs[:remaining]
        if not xs:
            break
        ys = [wrapped(x) for x in xs]
        if len(xs) == es.popsize:
            es.tell(xs, ys)
        else:
            break
    return wrapped.best, wrapped.used


def _scipy_de(objective, dimension, bounds, budget, seed):
    lo, hi = bounds
    wrapped = BudgetedObjective(objective, budget)
    popsize = 10
    population = popsize * dimension
    if budget < population:
        popsize = max(1, budget // dimension)
        population = popsize * dimension
    maxiter = max(0, budget // max(1, population) - 1)
    try:
        differential_evolution(
            wrapped,
            [(lo, hi)] * dimension,
            seed=seed,
            popsize=popsize,
            maxiter=maxiter,
            polish=False,
            updating="immediate",
            workers=1,
            tol=0.0,
            atol=0.0,
        )
    except BudgetExhausted:
        pass
    return wrapped.best, wrapped.used


def _scipy_local(method: str):
    def runner(objective, dimension, bounds, budget, seed):
        lo, hi = bounds
        wrapped = BudgetedObjective(objective, budget)
        x0 = _seeded_start(dimension, bounds, seed)
        options = {"maxfev": budget}
        if method == "Nelder-Mead":
            options.update({"xatol": 0.0, "fatol": 0.0, "maxiter": budget * 2})
        elif method == "Powell":
            options.update({"xtol": 0.0, "ftol": 0.0, "maxiter": budget * 2})
        try:
            minimize(
                wrapped,
                x0,
                method=method,
                bounds=[(lo, hi)] * dimension,
                options=options,
            )
        except BudgetExhausted:
            pass
        return wrapped.best, wrapped.used

    return runner


METHODS = {
    "RO-Hybrid-Multires": _resolutive,
    "CMA-ES(pycma)": _cma_es,
    "DE(scipy)": _scipy_de,
    "Nelder-Mead(scipy)": _scipy_local("Nelder-Mead"),
    "Powell(scipy)": _scipy_local("Powell"),
}


def default_budget(dimension: int) -> int:
    return max(1000, 300 * dimension)


def run(
    *,
    dimensions: list[int],
    seeds: int,
    raw_output: Path,
    summary_output: Path,
    budget_override: int | None = None,
) -> None:
    rows: list[dict[str, object]] = []
    for dimension in dimensions:
        budget = budget_override if budget_override is not None else default_budget(dimension)
        for benchmark_name, (objective, bounds) in DEFAULT_BENCHMARKS.items():
            for seed in range(seeds):
                for optimizer_name, runner in METHODS.items():
                    value, used = runner(objective, dimension, bounds, budget, seed)
                    rows.append(
                        {
                            "benchmark": benchmark_name,
                            "dimension": dimension,
                            "budget": budget,
                            "seed": seed,
                            "optimizer": optimizer_name,
                            "fun": value,
                            "evaluations": used,
                        }
                    )

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    with raw_output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, int, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["benchmark"]), int(row["dimension"]), int(row["budget"]), str(row["optimizer"]))
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for (benchmark, dimension, budget, optimizer), group in grouped.items():
        values = np.asarray([float(row["fun"]) for row in group], dtype=float)
        evals = np.asarray([int(row["evaluations"]) for row in group], dtype=int)
        summary.append(
            {
                "benchmark": benchmark,
                "dimension": dimension,
                "budget": budget,
                "optimizer": optimizer,
                "seeds": len(group),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "best": float(np.min(values)),
                "worst": float(np.max(values)),
                "median_evaluations": float(np.median(evals)),
                "max_evaluations": int(np.max(evals)),
            }
        )

    summary.sort(key=lambda row: (str(row["benchmark"]), int(row["dimension"]), float(row["median"])))
    with summary_output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 10, 30])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget", type=int, default=None, help="optional same budget for every dimension")
    parser.add_argument("--raw-output", type=Path, default=Path("results/release-benchmark-v1/raw.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/release-benchmark-v1/summary.csv"))
    args = parser.parse_args()
    run(
        dimensions=args.dimensions,
        seeds=args.seeds,
        raw_output=args.raw_output,
        summary_output=args.summary_output,
        budget_override=args.budget,
    )
