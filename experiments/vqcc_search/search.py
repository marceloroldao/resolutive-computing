"""Equal-budget precision search: RO-V2, random search and differential evolution."""
from __future__ import annotations
import dataclasses
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "vqcc"))
from resolutive.optimization.v2 import ResolutiveV2
from resolutive import optimize

LOW = np.array([11, 14, 25, 25, 26])
HIGH = np.array([18, 19, 31, 31, 32])
INCUMBENT = (14, 17, 28, 28, 29)
BUDGET = 1024
SEEDS = tuple(range(10))
METHODS = ("RO-V2", "RO-auto", "random", "differential_evolution")
N = 3133452

def decode(point):
    return tuple(int(v) for v in np.rint(LOW + np.clip(point, 0, 1) * (HIGH - LOW)))

def encode(config):
    return (np.array(config, dtype=float) - LOW) / (HIGH - LOW)

class BudgetEnd(Exception):
    pass

def optimize_one(method, seed, evaluate, budget=BUDGET):
    """All calls, including repeats and the shared incumbent, count toward budget."""
    start = time.monotonic()
    calls = 0
    best = None
    distinct = set()
    def objective(point):
        nonlocal calls, best
        if calls >= budget:
            raise BudgetEnd
        calls += 1
        config = decode(point)
        distinct.add(config)
        row = evaluate(config)
        if best is None or row["objective"] < best["objective"]:
            best = dict(row)
        return row["objective"]
    objective(encode(INCUMBENT))
    remaining = budget - calls
    rng = np.random.default_rng(seed)
    try:
        if method == "RO-V2":
            ResolutiveV2().minimize(objective, dimension=5, bounds=(0.0, 1.0),
                                   budget=remaining, seed=seed)
        elif method == "RO-auto":
            optimize(objective, dimension=5, bounds=(0.0, 1.0),
                     budget=remaining, seed=seed, mode="auto")
        elif method == "random":
            while calls < budget:
                objective(rng.random(5))
        elif method == "differential_evolution":
            differential_evolution(objective, [(0.0, 1.0)] * 5, rng=rng,
                                   popsize=7, maxiter=1000, polish=False,
                                   tol=0.0, atol=0.0, updating="immediate")
        else:
            raise ValueError(method)
    except BudgetEnd:
        pass
    # RO may stop with a partial generation or coordinate sweep. Give every
    # method its full allocation; this common padding policy is predeclared.
    while calls < budget:
        objective(rng.random(5))
    assert calls == budget
    return dict(method=method, seed=seed, calls=calls, distinct_configurations=len(distinct),
                search_seconds=time.monotonic()-start, selected=best)

def training_inputs():
    rng = np.random.default_rng(20261116)
    points = {1, 2, 3, N, 2223871, 2227676, 2227071, 2588411}
    points.update(int(x) for x in rng.integers(1, N+1, size=1024))
    # Every representable boundary and its neighbours, across all scales.
    for k in range(11):
        for j in range(16):
            center = (4**k) * (1 + j/4)
            for offset in (-1, 0, 1):
                x = int(center) + offset
                if 1 <= x <= N:
                    points.add(x)
    return np.array(sorted(points), dtype=np.int64)

def score_from_margin(structure, margin, verifier):
    if margin <= 0:
        return math.inf
    rotations = structure["rotations"]
    eps = math.pi * margin / 2**64 / rotations if rotations else None
    t_cost = structure["t_gates"] + (rotations * verifier.rotation_t_count(eps) if rotations else 0)
    return structure["qubits"] * (structure["toffolis"] + t_cost / 4)

def phase_errors(config, xs, candidate, ref, verifier):
    precision = ref.Precision(*config)
    ys = candidate.model(precision, xs)
    phase = ys.astype(np.uint64) << np.uint64(64-precision.output_bits(10))
    target, tolerance = verifier.reference_values(verifier.ProblemSpecInvSqrt.for_n_p(10),
                                                 xs.astype(np.uint64))
    with np.errstate(over="ignore"):
        errors = (phase-target).view(np.int64)
    used = np.abs(errors).view(np.uint64)
    bad = used > tolerance
    ratios = used.astype(float) / tolerance.astype(float)
    # Avoid unsigned underflow on failed inputs.
    margin = min(int(t)-int(e) for t,e in zip(tolerance,used))
    return int(bad.sum()), float(ratios.max()), int(xs[int(np.argmax(ratios))]), margin

def main():
    import run as bridge
    out = Path(os.environ.get("VQCC_OUTPUT", "vqcc-search-results")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bridge.OUT, bridge.VENDOR = out, out / "upstream"
    bridge.fetch()
    import candidate
    import nr_reference_circuit as ref
    import verify_submission_invsqrt as verifier

    def write(name, value):
        (out/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")

    structure_cache = {}
    def structure(config):
        if config not in structure_cache:
            circuit = candidate.build(ref.Precision(*config))
            stats = circuit.operation_statistics()
            structure_cache[config] = dict(
                qubits=int(circuit.num_qubits), toffolis=int(stats.num_toffoli_like),
                t_gates=int(stats.num_t_like),
                rotations=int(stats.num_z_pow(min_clifford_hierarchy_level=4)))
        return structure_cache[config]

    xs = training_inputs()
    # Frozen targets can be precomputed once; don't change the sampled fitness.
    target_cache = verifier.reference_values(verifier.ProblemSpecInvSqrt.for_n_p(10),
                                             xs.astype(np.uint64))
    sampled = {}
    def evaluate(config):
        if config in sampled:
            return sampled[config]
        row = dict(configuration=list(config))
        try:
            precision = ref.Precision(*config)
            ys = candidate.model(precision, xs)
            phase = ys.astype(np.uint64) << np.uint64(64-precision.output_bits(10))
            target, tolerance = target_cache
            with np.errstate(over="ignore"):
                errors = (phase-target).view(np.int64)
            used = np.abs(errors).view(np.uint64)
            ratio = float((used.astype(float)/tolerance).max())
            failures = int((used > tolerance).sum())
            margin = min(int(t)-int(e) for t,e in zip(tolerance,used))
            row.update(sample_failures=failures, sample_worst_ratio=ratio)
            if failures or margin <= 0:
                row["objective"] = 1e10 + min(ratio, 1e6)*1e6
                row["sample_feasible"] = False
            else:
                row.update(objective=score_from_margin(structure(config),margin,verifier),
                           sample_feasible=True)
        except (OverflowError, ValueError):
            row.update(objective=1e15, sample_feasible=False, model_overflow=True)
        sampled[config] = row
        return row

    report = dict(status="SEARCHING", protocol=dict(
        methods=list(METHODS), seeds=list(SEEDS), budget_per_method_seed=BUDGET,
        includes_common_incumbent=True, duplicate_calls_count=True,
        training_points=len(xs), bounds=[LOW.tolist(),HIGH.tolist()],
        representation="continuous [0,1]^5 rounded to precision integers",
        objective="sampled Q*C including rotation synthesis; sampled margin can be optimistic",
        validation="one training-selected configuration per method/seed; full domain; no feedback",
        clocks="cache is shared; wall times cannot support comparative speed claims"),
        runs=[], full_model_results=[], native_verification=None,
        new_physics_tested=False, leaderboard_rank_established=False)
    write("search-report.json", report)
    for seed in SEEDS:
        for method in METHODS:
            row = optimize_one(method,seed,evaluate)
            report["runs"].append(row)
            write("search-report.json", report)
            print("search",json.dumps(row),flush=True)

    full_cache = {}
    def full(config):
        if config in full_cache:
            return full_cache[config]
        failures, worst, worst_x, margin = 0, 0.0, 0, 1<<64
        for lo in range(1,N+1,65536):
            points = np.arange(lo,min(N+1,lo+65536),dtype=np.int64)
            bad, ratio, x, m = phase_errors(config,points,candidate,ref,verifier)
            failures += bad
            margin = min(margin,m)
            if ratio > worst:
                worst, worst_x = ratio,x
        row = dict(configuration=list(config), inputs=N, failures=failures,
                   worst_error_over_tolerance=worst,worst_x=worst_x)
        if failures == 0 and margin > 0:
            row.update(full_model_cost=score_from_margin(structure(config),margin,verifier),
                       structure=structure(config),minimum_margin_uint64=margin)
        else:
            row["full_model_cost"] = None
        full_cache[config]=row
        report["full_model_results"].append(row)
        write("search-report.json",report)
        print("full_model",json.dumps(row),flush=True)
        return row

    incumbent = full(INCUMBENT)
    previous = json.loads((ROOT/"reports/vqcc/run-43990346433.json").read_text())
    prior_cost = previous["candidate"]["score"]
    if abs(incumbent["full_model_cost"]-prior_cost) > 1e-6:
        raise RuntimeError("Full integer model did not reproduce prior native cost")

    for row in report["runs"]:
        validated = full(tuple(row["selected"]["configuration"]))
        row["selected_full_model"] = validated
        row["retained_cost"] = min(prior_cost, validated["full_model_cost"]
                                  if validated["full_model_cost"] is not None else prior_cost)
    report["statistics"] = {}
    for method in METHODS:
        rows=[r for r in report["runs"] if r["method"]==method]
        costs=np.array([r["retained_cost"] for r in rows])
        report["statistics"][method]=dict(
            median=float(np.median(costs)),mean=float(np.mean(costs)),
            std=float(np.std(costs)),best=float(costs.min()),worst=float(costs.max()),
            selected_full_domain_failures=sum(r["selected_full_model"]["failures"]>0 for r in rows))
    report["paired_ro_wins"]={}
    for ro,other in ((ro,other) for ro in METHODS[:2] for other in METHODS[2:]):
        pairs=[(
            next(r["retained_cost"] for r in report["runs"] if r["seed"]==s and r["method"]==ro),
            next(r["retained_cost"] for r in report["runs"] if r["seed"]==s and r["method"]==other))
            for s in SEEDS]
        report["paired_ro_wins"][ro+" vs "+other]=dict(wins=sum(a<b-1e-6 for a,b in pairs),
                                            ties=sum(abs(a-b)<=1e-6 for a,b in pairs),
                                            losses=sum(a>b+1e-6 for a,b in pairs))
    eligible=[r for r in full_cache.values() if r["full_model_cost"] is not None]
    best=min(eligible,key=lambda r:r["full_model_cost"])
    report.update(status="SEARCH_AND_FULL_MODEL_COMPLETE",incumbent_cost=prior_cost,
                  best_full_model=best,unique_sampled_configurations=len(sampled))
    write("search-report.json",report)
    if best["full_model_cost"] < prior_cost-1e-6:
        config=tuple(best["configuration"])
        path=out/"search-winner.kmb"
        candidate.build(ref.Precision(*config)).to_file(str(path),format="kmb")
        print("native_verification_start",json.dumps(best),flush=True)
        result=bridge.verify_file("search-winner",path)
        report["native_verification"]=result
        if result.get("status")!="PASS":
            report["status"]="SEARCH_WINNER_NATIVE_FAILED"
        elif abs(result["score"]-best["full_model_cost"]) > 1e-6:
            report["status"]="NATIVE_MODEL_COST_MISMATCH"
        else:
            report["status"]="SEARCH_WINNER_FULL_NATIVE_PASS"
    else:
        report["status"]="INCUMBENT_RETAINED_NO_NEW_NATIVE_RUN_NEEDED"
    write("search-report.json",report)
    summary="# RO precision-search comparison\n\n"
    summary+="Status: "+report["status"]+"\n\n"
    summary+="| Method | Median retained full-model cost | Best | Worst | Invalid selections |\n|---|---:|---:|---:|---:|\n"
    for method,s in report["statistics"].items():
        summary+=f'| {method} | {s["median"]:.3f} | {s["best"]:.3f} | {s["worst"]:.3f} | {s["selected_full_domain_failures"]} |\n'
    summary+="\nSearch cost statistics use exact full-domain integer modelling. Only the final winner receives a fresh 64-shot native verifier run. The incumbent was previously native-verified.\n"
    (out/"SUMMARY.md").write_text(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"],"a") as f:
            f.write(summary)
    print("FINAL_REPORT",json.dumps(report),flush=True)
    if report["status"] in ("SEARCH_WINNER_NATIVE_FAILED","NATIVE_MODEL_COST_MISMATCH"):
        raise SystemExit(1)

if __name__ == "__main__":
    main()

