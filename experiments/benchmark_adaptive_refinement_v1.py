"""Benchmark adaptive refinement against current Hybrid-Multires baseline."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from resolutive.benchmarks.functions import DEFAULT_BENCHMARKS
from resolutive.optimization.hybrid_multires import ResolutiveHybridMultiResolution
from adaptive_refinement_v1 import ResolutiveAdaptiveRefinementV1


def default_budget(dimension: int) -> int:
    return max(1000, 300 * dimension)


def run(dimensions: list[int], seeds: int, raw_output: Path, summary_output: Path) -> None:
    methods = {
        "RO-Hybrid-Multires": ResolutiveHybridMultiResolution(),
        "RO-Adaptive-Refinement-v1-exp": ResolutiveAdaptiveRefinementV1(),
    }
    rows = []
    for dimension in dimensions:
        budget = default_budget(dimension)
        for benchmark_name, (objective, bounds) in DEFAULT_BENCHMARKS.items():
            for seed in range(seeds):
                for method_name, method in methods.items():
                    result = method.minimize(objective, dimension=dimension, bounds=bounds, budget=budget, seed=seed)
                    rows.append({"benchmark": benchmark_name, "dimension": dimension, "budget": budget, "seed": seed, "optimizer": method_name, "fun": float(result.fun), "evaluations": int(result.evaluations)})

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    with raw_output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    grouped = {}
    for row in rows:
        key = (row["benchmark"], row["dimension"], row["budget"], row["optimizer"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (benchmark, dimension, budget, optimizer), group in grouped.items():
        vals = np.asarray([r["fun"] for r in group], dtype=float)
        evals = np.asarray([r["evaluations"] for r in group], dtype=int)
        summary.append({"benchmark": benchmark, "dimension": dimension, "budget": budget, "optimizer": optimizer, "seeds": len(group), "median": float(np.median(vals)), "mean": float(np.mean(vals)), "best": float(np.min(vals)), "worst": float(np.max(vals)), "median_evaluations": float(np.median(evals))})
    summary.sort(key=lambda r: (r["benchmark"], r["dimension"], r["median"]))
    with summary_output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)

    cases = {}
    for row in summary:
        cases.setdefault((row["benchmark"], row["dimension"]), {})[row["optimizer"]] = row["median"]
    wins = ties = losses = 0
    for (benchmark, dimension), values in sorted(cases.items()):
        base = values["RO-Hybrid-Multires"]
        cand = values["RO-Adaptive-Refinement-v1-exp"]
        tol = 1e-12 + 1e-9 * max(abs(base), abs(cand), 1.0)
        if cand < base - tol:
            wins += 1; verdict = "WIN"
        elif cand > base + tol:
            losses += 1; verdict = "LOSS"
        else:
            ties += 1; verdict = "TIE"
        print(f"{benchmark} {dimension}D: {verdict} baseline={base:.12g} candidate={cand:.12g}")
    print(f"ADAPTIVE_V1_WTL wins={wins} ties={ties} losses={losses}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dimensions", type=int, nargs="+", default=[2, 10, 30])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--raw-output", type=Path, default=Path("results/adaptive-refinement-v1/raw.csv"))
    p.add_argument("--summary-output", type=Path, default=Path("results/adaptive-refinement-v1/summary.csv"))
    a = p.parse_args()
    run(a.dimensions, a.seeds, a.raw_output, a.summary_output)
