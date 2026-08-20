from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from resolutive.benchmarks.functions import ackley, rastrigin, rosenbrock
from resolutive.optimization.v2 import ResolutiveV2
from resolutive.optimization.v5 import ResolutiveV5
from resolutive.specialist_sessions import V2Session, V5Session

FUNCTIONS = {
    "ackley": (ackley, (-32.768, 32.768)),
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-2.048, 2.048)),
}


def run_session(session, objective):
    while not session.done:
        try:
            batch = session.ask()
        except RuntimeError as exc:
            if "complete" in str(exc):
                break
            raise
        values = np.asarray([objective(x) for x in batch.points], dtype=float)
        session.tell(values)
    return session.result()


def run(*, dimension: int, budget: int, seeds: int, output: str) -> None:
    rows = []
    for function_name, (objective, bounds) in FUNCTIONS.items():
        for seed in range(seeds):
            pairs = [
                (
                    "v2",
                    ResolutiveV2().minimize(
                        objective, dimension=dimension, bounds=bounds, budget=budget, seed=seed
                    ),
                    run_session(
                        V2Session(
                            dimension=dimension, bounds=bounds, budget=budget, seed=seed
                        ),
                        objective,
                    ),
                ),
                (
                    "v5_no_collapse",
                    ResolutiveV5(use_collapse=False).minimize(
                        objective, dimension=dimension, bounds=bounds, budget=budget, seed=seed
                    ),
                    run_session(
                        V5Session(
                            dimension=dimension,
                            bounds=bounds,
                            budget=budget,
                            seed=seed,
                            use_collapse=False,
                        ),
                        objective,
                    ),
                ),
            ]
            for engine, mono, inc in pairs:
                abs_gap = abs(float(inc.fun) - float(mono.fun))
                denom = max(abs(float(mono.fun)), 1e-15)
                rows.append({
                    "function": function_name,
                    "engine": engine,
                    "seed": seed,
                    "dimension": dimension,
                    "budget": budget,
                    "monolithic_fun": mono.fun,
                    "incremental_fun": inc.fun,
                    "absolute_gap": abs_gap,
                    "relative_gap": abs_gap / denom,
                    "monolithic_evaluations": mono.evaluations,
                    "incremental_evaluations": inc.evaluations,
                })

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", default="results/specialist_session_compare.csv")
    args = parser.parse_args()
    run(dimension=args.dimension, budget=args.budget, seeds=args.seeds, output=args.output)
