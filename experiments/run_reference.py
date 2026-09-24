"""Run a versioned Resolutive Computing reference experiment.

The runner reads a JSON experiment specification, validates it, executes all
configured benchmarks/optimizers/dimensions/seeds under the declared evaluation
budgets, and writes raw results plus a metadata manifest and aggregated summary.
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

REQUIRED_KEYS = {
    "schema_version",
    "experiment_id",
    "status",
    "rsms_compatibility",
    "benchmarks",
    "optimizers",
    "dimensions",
    "classical_budget",
    "scaling_budget_per_dimension",
    "seeds",
    "seed_policy",
    "metrics",
    "claim_policy",
}


def validate_config(config: dict[str, object]) -> None:
    missing = REQUIRED_KEYS.difference(config)
    if missing:
        raise ValueError(f"missing required experiment keys: {sorted(missing)}")

    if int(config["schema_version"]) != 1:
        raise ValueError("unsupported schema_version; expected 1")
    if not str(config["experiment_id"]).strip():
        raise ValueError("experiment_id must be non-empty")
    if int(config["seeds"]) < 1:
        raise ValueError("seeds must be >= 1")
    if int(config["classical_budget"]) < 1:
        raise ValueError("classical_budget must be >= 1")
    if int(config["scaling_budget_per_dimension"]) < 1:
        raise ValueError("scaling_budget_per_dimension must be >= 1")

    dimensions = list(config["dimensions"])
    if not dimensions or any(int(d) < 1 for d in dimensions):
        raise ValueError("dimensions must contain positive integers")

    benchmarks = list(config["benchmarks"])
    if not benchmarks:
        raise ValueError("benchmarks must not be empty")
    unknown_benchmarks = sorted(set(benchmarks).difference(DEFAULT_BENCHMARKS))
    if unknown_benchmarks:
        raise ValueError(f"unknown benchmarks: {unknown_benchmarks}")

    optimizers = list(config["optimizers"])
    if not optimizers:
        raise ValueError("optimizers must not be empty")
    unknown_optimizers = sorted(set(optimizers).difference(OPTIMIZERS))
    if unknown_optimizers:
        raise ValueError(f"unknown optimizers: {unknown_optimizers}")

    expected_seed_policy = f"integer seeds 0..{int(config['seeds']) - 1} inclusive"
    if str(config["seed_policy"]) != expected_seed_policy:
        raise ValueError(
            f"seed_policy inconsistent with seeds; expected {expected_seed_policy!r}"
        )


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
    validate_config(config)
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
