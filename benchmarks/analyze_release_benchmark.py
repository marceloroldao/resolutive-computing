"""Analyze Resolutive Computing v1 release benchmark summary.

Produces per-scenario ranks plus overall win/tie/loss counts for the primary
Resolutive optimizer versus each external baseline. Lower median objective
value is better.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

PRIMARY = "RO-Hybrid-Multires"


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def analyze(summary_path: Path, output_path: Path, *, rtol: float = 1e-9, atol: float = 1e-12) -> None:
    rows = _load(summary_path)
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["benchmark"], int(row["dimension"]), int(row["budget"]))
        grouped[key].append(row)

    output: list[dict[str, object]] = []
    head_to_head: dict[str, dict[str, int]] = defaultdict(lambda: {"wins": 0, "ties": 0, "losses": 0})

    for (benchmark, dimension, budget), group in sorted(grouped.items()):
        medians = {row["optimizer"]: float(row["median"]) for row in group}
        if PRIMARY not in medians:
            raise ValueError(f"missing {PRIMARY} in {benchmark} {dimension}D")

        ordered = sorted(medians.items(), key=lambda item: item[1])
        primary_value = medians[PRIMARY]
        primary_rank = 1 + sum(value < primary_value and not np.isclose(value, primary_value, rtol=rtol, atol=atol) for _, value in ordered)

        output.append({
            "benchmark": benchmark,
            "dimension": dimension,
            "budget": budget,
            "primary_median": primary_value,
            "primary_rank": primary_rank,
            "best_optimizer": ordered[0][0],
            "best_median": ordered[0][1],
        })

        for optimizer, value in medians.items():
            if optimizer == PRIMARY:
                continue
            if np.isclose(primary_value, value, rtol=rtol, atol=atol):
                head_to_head[optimizer]["ties"] += 1
            elif primary_value < value:
                head_to_head[optimizer]["wins"] += 1
            else:
                head_to_head[optimizer]["losses"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        fields = ["benchmark", "dimension", "budget", "primary_median", "primary_rank", "best_optimizer", "best_median"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)

    print(f"scenarios={len(output)}")
    if output:
        mean_rank = float(np.mean([float(row["primary_rank"]) for row in output]))
        firsts = sum(int(row["primary_rank"]) == 1 for row in output)
        print(f"primary={PRIMARY} mean_rank={mean_rank:.3f} first_place={firsts}/{len(output)}")
    for optimizer in sorted(head_to_head):
        record = head_to_head[optimizer]
        print(f"vs {optimizer}: W={record['wins']} T={record['ties']} L={record['losses']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("results/release-benchmark-v1/summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/release-benchmark-v1/ranking.csv"))
    args = parser.parse_args()
    analyze(args.summary, args.output)
