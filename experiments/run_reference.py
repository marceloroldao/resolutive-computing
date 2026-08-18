"""Run a versioned Resolutive Computing reference experiment.

The runner reads a JSON experiment specification, executes all configured
benchmarks/optimizers/dimensions/seeds under the declared evaluation budgets,
and writes raw results plus a metadata manifest and aggregated summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from resolutive.benchmarks.functions import DEFAULT_BENCHMARKS
from resolutive.optimization.baselines import DifferentialEvolution, RandomSearch, SimulatedAnnealing
from resolutive.optimization.v2 import ResolutiveV2
from resolutive.optimization.v5 import ResolutiveV5

OPTIMIZERS = {
    "RandomSearch": RandomSearch,
    "SimulatedAnnealing": SimulatedAnnealing,
    "DifferentialEvolution": DifferentialEvolution,
    "RO-V2": ResolutiveV2,
    "RO-V5": ResolutiveV5,
}


def _git_sha() -> str:
    env_sha = os.environ.get("GITHUB_SHA")
    if env_sha:
        return env_sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run(config_path: Path, output_dir: Path, smoke: bool = False) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_id = config["experiment_id"]
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []
    seeds = 1 if smoke else int(config["seeds"])
    dimensions = [config["dimensions"][0]] if smoke else list(config["dimensions"])
    optimizers = list(config["optimizers"])
    if smoke:
        optimizers = [name for name in optimizers if name in {"DifferentialEvolution", "RO-V2", "RO-V5"}]

    for dimension in dimensions:
        budget = min(300, int(config["classical_budget"])) if smoke else int(config["scaling_budget_per_dimension"]) * int(dimension)
        for benchmark_name in config["benchmarks"]:
            objective, bounds = DEFAULT_BENCHMARKS[benchmark_name]
            for optimizer_name in optimizers:
                optimizer = OPTIMIZERS[optimizer_name]()
                for seed in range(seeds):
                    result = optimizer.minimize(
                        objective,
                        dimension=int(dimension),
                        bounds=bounds,
                        budget=int(budget),
                        seed=seed,
                    )
                    raw_rows.append({
                        "experiment_id": experiment_id,
                        "commit": _git_sha(),
                        "benchmark": benchmark_name,
                        "optimizer": optimizer_name,
                        "dimension": int(dimension),
                        "budget": int(budget),
                        "seed": seed,
                        "value": float(result.fun),
                        "evaluations": int(result.evaluations),
                    })

    raw_path = output_dir / "raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_rows[0].keys())
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_rows: list[dict[str, object]] = []
    keys = sorted({(r["benchmark"], r["optimizer"], r["dimension"], r["budget"]) for r in raw_rows})
    for benchmark_name, optimizer_name, dimension, budget in keys:
        subset = [r for r in raw_rows if (r["benchmark"], r["optimizer"], r["dimension"], r["budget"]) == (benchmark_name, optimizer_name, dimension, budget)]
        values = np.array([r["value"] for r in subset], dtype=float)
        evaluations = np.array([r["evaluations"] for r in subset], dtype=int)
        summary_rows.append({
            "benchmark": benchmark_name,
            "optimizer": optimizer_name,
            "dimension": dimension,
            "budget": budget,
            "seeds": len(subset),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "best": float(np.min(values)),
            "worst": float(np.max(values)),
            "max_evaluations_used": int(np.max(evaluations)),
        })

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = {
        "experiment_id": experiment_id,
        "schema_version": config["schema_version"],
        "status": config["status"],
        "rsms_compatibility": config["rsms_compatibility"],
        "commit": _git_sha(),
        "config_path": str(config_path),
        "smoke": smoke,
        "raw_rows": len(raw_rows),
        "summary_rows": len(summary_rows),
        "claim_policy": config["claim_policy"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/reference_v0.1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reference-v0.1"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output_dir, smoke=args.smoke)
