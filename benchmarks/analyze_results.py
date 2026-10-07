"""Summarize benchmark CSV files into objective rankings.

The input is the aggregated CSV produced by the benchmark runners. Rankings are
computed independently per benchmark and dimension using the median objective
value (lower is better), then averaged across cases.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def analyze(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input CSV is empty")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row.get("dimension", ""))].append(row)

    ranks: dict[str, list[float]] = defaultdict(list)
    wins: dict[str, int] = defaultdict(int)
    ratios: dict[str, list[float]] = defaultdict(list)

    for _, case_rows in grouped.items():
        ordered = sorted(case_rows, key=lambda row: float(row["median"]))
        best = float(ordered[0]["median"])
        for rank, row in enumerate(ordered, start=1):
            name = row["optimizer"]
            value = float(row["median"])
            ranks[name].append(float(rank))
            if rank == 1:
                wins[name] += 1
            if abs(best) > 1e-15:
                ratios[name].append(value / best)
            else:
                ratios[name].append(1.0 if abs(value) <= 1e-15 else float("inf"))

    summary = []
    for name in sorted(ranks):
        finite_ratios = [r for r in ratios[name] if r != float("inf")]
        mean_ratio = sum(finite_ratios) / len(finite_ratios) if finite_ratios else float("inf")
        summary.append({
            "optimizer": name,
            "cases": len(ranks[name]),
            "wins": wins[name],
            "mean_rank": sum(ranks[name]) / len(ranks[name]),
            "mean_ratio_to_best": mean_ratio,
        })
    summary.sort(key=lambda row: (float(row["mean_rank"]), -int(row["wins"])))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input, args.output)
