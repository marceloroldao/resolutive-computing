"""Benchmark regime-gated adaptive refinement v2 against current baseline."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from resolutive.benchmarks.functions import DEFAULT_BENCHMARKS
from resolutive.optimization.hybrid_multires import ResolutiveHybridMultiResolution
from adaptive_refinement_v2 import ResolutiveAdaptiveRefinementV2


def default_budget(dimension: int) -> int:
    return max(1000, 300 * dimension)


def run(dimensions: list[int], seeds: int, raw_output: Path, summary_output: Path) -> None:
    methods = {
        "RO-Hybrid-Multires": ResolutiveHybridMultiResolution(),
        "RO-Adaptive-Refinement-v2-exp": ResolutiveAdaptiveRefinementV2(),
    }
    rows = []
    for dimension in dimensions:
        budget = default_budget(dimension)
        for benchmark_name, (objective, bounds) in DEFAULT_BENCHMARKS.items():
            for seed in range(seeds):
                for method_name, method in methods.items():
                    result = method.minimize(objective, dimension=dimension, bounds=bounds, budget=budget, seed=seed)
                    diag = getattr(result, "diagnostics", {}) or {}
                    rows.append({
                        "benchmark": benchmark_name,
                        "dimension": dimension,
                        "budget": budget,
                        "seed": seed,
                        "optimizer": method_name,
                        "fun": float(result.fun),
                        "evaluations": int(result.evaluations),
                        "local_regime": diag.get("local_regime", ""),
                        "coarse_moves": diag.get("coarse_moves", ""),
                        "fine_moves": diag.get("fine_moves", ""),
                        "coarse_gain": diag.get("coarse_gain", ""),
                        "fine_gain": diag.get("fine_gain", ""),
                    })

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
        local_flags = [r["local_regime"] for r in group if r["local_regime"] != ""]
        summary.append({
            "benchmark": benchmark,
            "dimension": dimension,
            "budget": budget,
            "optimizer": optimizer,
            "seeds": len(group),
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "best": float(np.min(vals)),
            "worst": float(np.max(vals)),
            "median_evaluations": float(np.median(evals)),
            "local_regime_rate": float(np.mean([bool(v) for v in local_flags])) if local_flags else "",
        })
    summary.sort(key=lambda r: (r["benchmark"], r["dimension"], r["median"]))
    with summary_output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)

    cases = {}
    for row in summary:
        cases.setdefault((row["benchmark"], row["dimension"]), {})[row["optimizer"]] = row
    wins = ties = losses = 0
    for (benchmark, dimension), values in sorted(cases.items()):
        base = values["RO-Hybrid-Multires"]["median"]
        cand_row = values["RO-Adaptive-Refinement-v2-exp"]
        cand = cand_row["median"]
        tol = 1e-12 + 1e-9 * max(abs(base), abs(cand), 1.0)
        if cand < base - tol:
            wins += 1; verdict = "WIN"
        elif cand > base + tol:
            losses += 1; verdict = "LOSS"
        else:
            ties += 1; verdict = "TIE"
        print(f"{benchmark} {dimension}D: {verdict} baseline={base:.12g} candidate={cand:.12g} local_rate={cand_row['local_regime_rate']}")
    print(f"ADAPTIVE_V2_WTL wins={wins} ties={ties} losses={losses}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dimensions", type=int, nargs="+", default=[2, 10, 30])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--raw-output", type=Path, default=Path("results/adaptive-refinement-v2/raw.csv"))
    p.add_argument("--summary-output", type=Path, default=Path("results/adaptive-refinement-v2/summary.csv"))
    a = p.parse_args()
    run(a.dimensions, a.seeds, a.raw_output, a.summary_output)
