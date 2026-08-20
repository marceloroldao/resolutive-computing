from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from resolutive import create_session
from resolutive.benchmarks.functions import ackley, rastrigin, rosenbrock
from resolutive.optimization.hybrid_multires import ResolutiveHybridMultiResolution


FUNCTIONS = {
    "ackley": (ackley, (-32.768, 32.768)),
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-2.048, 2.048)),
}


def run_session(objective, *, dimension: int, bounds: tuple[float, float], budget: int, seed: int):
    session = create_session(
        dimension=dimension,
        bounds=bounds,
        budget=budget,
        seed=seed,
        mode="multires",
    )
    while not session.done:
        batch = session.ask()
        values = np.asarray([objective(x) for x in batch.points], dtype=float)
        session.tell(values)
    return session.result()


def run(*, dimension: int, budget: int, seeds: int, output: str) -> None:
    rows: list[dict[str, object]] = []
    for name, (objective, bounds) in FUNCTIONS.items():
        for seed in range(seeds):
            mono = ResolutiveHybridMultiResolution().minimize(
                objective,
                dimension=dimension,
                bounds=bounds,
                budget=budget,
                seed=seed,
            )
            inc = run_session(
                objective,
                dimension=dimension,
                bounds=bounds,
                budget=budget,
                seed=seed,
            )
            rows.append({
                "function": name,
                "seed": seed,
                "dimension": dimension,
                "budget": budget,
                "monolithic_fun": mono.fun,
                "incremental_fun": inc.fun,
                "monolithic_evaluations": mono.evaluations,
                "incremental_evaluations": inc.evaluations,
                "ratio_incremental_over_monolithic": inc.fun / max(abs(mono.fun), 1e-15),
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
    parser.add_argument("--output", default="results/multires_session_compare.csv")
    args = parser.parse_args()
    run(dimension=args.dimension, budget=args.budget, seeds=args.seeds, output=args.output)
