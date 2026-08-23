"""Historical RO-V4 adapted to the common optimizer interface.

Source basis: user-provided otimizador_resolutivo_v4.py. The search modes,
success adaptation, migration, and anisotropic collapse are preserved. The
adapter enforces the repository's strict evaluation-budget contract.
"""
from __future__ import annotations

import numpy as np

from .common import Objective, OptimizationResult, validate_bounds


class ResolutiveV4Historical:
    def __init__(self, population: int = 40, groups: int = 4) -> None:
        if population < 16 or groups < 2 or population % groups:
            raise ValueError("population must be divisible by groups and >= 16")
        self.population = population
        self.groups = groups

    def minimize(self, objective: Objective, *, dimension: int,
                 bounds: tuple[float, float], budget: int = 6000,
                 seed: int = 0) -> OptimizationResult:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        lo, hi = validate_bounds(bounds)
        span = hi - lo
        pop, k = self.population, self.groups
        per = pop // k
        if budget <= pop:
            raise ValueError("budget must exceed population")
        rng = np.random.default_rng(seed)
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget:
                return None
            used += 1
            return float(objective(x))

        groups = []
        for _ in range(k):
            X = rng.uniform(lo, hi, (per, dimension))
            vals = np.array([evaluate(x) for x in X], dtype=float)
            groups.append({"X": X, "vals": vals, "sigma": 0.16 * span,
                           "stall": 0, "mode": "multi", "success_hist": []})

        bestv = float("inf")
        best = None
        global_hist = []
        golden = 2 * np.pi * (1 - 1 / ((1 + np.sqrt(5)) / 2))
        gen = 0

        while used + pop <= budget:
            gen += 1
            allX = np.vstack([g["X"] for g in groups])
            allV = np.concatenate([g["vals"] for g in groups])
            jg = int(np.argmin(allV))
            if allV[jg] < bestv:
                bestv = float(allV[jg]); best = allX[jg].copy()
            global_hist.append(bestv)
            global_hist = global_hist[-12:]

            for gi, g in enumerate(groups):
                X, vals = g["X"], g["vals"]
                order = np.argsort(vals)
                elite = X[order[:max(3, per // 3)]]
                local_best = elite[0]
                C = np.cov(elite, rowvar=False) + np.eye(dimension) * (1e-10 * span ** 2)
                evals, evecs = np.linalg.eigh(C)
                anis = (evals[-1] + 1e-15) / (evals[0] + 1e-15)
                spread = np.sqrt(np.trace(C)) / span
                recent_impr = 0.0
                if len(global_hist) >= 4:
                    recent_impr = (global_hist[-4] - global_hist[-1]) / (abs(global_hist[-4]) + 1e-12)

                if anis > 80 and spread < 0.18:
                    mode = "valley"
                elif g["stall"] >= 6 or recent_impr < 1e-5:
                    mode = "multi"
                else:
                    mode = "single"

                Y = []
                for i, x in enumerate(X):
                    target = elite[rng.integers(len(elite))]
                    if mode == "single":
                        direction = 0.50 * (best - x) + 0.30 * (target - x)
                        a, b = rng.choice(dimension, 2, replace=False)
                        ang = golden * (gen + i + 1)
                        va, vb = direction[a], direction[b]
                        direction[a] = np.cos(ang) * va - np.sin(ang) * vb
                        direction[b] = np.sin(ang) * va + np.cos(ang) * vb
                        noise = rng.multivariate_normal(np.zeros(dimension), C)
                        noise = noise / (np.linalg.norm(noise) + 1e-12) * g["sigma"]
                        step = direction + noise
                    elif mode == "multi":
                        peer = elite[rng.integers(len(elite))]
                        noise = rng.normal(size=dimension); noise /= np.linalg.norm(noise) + 1e-12
                        step = 0.30 * (peer - x) + 0.20 * (local_best - x) + noise * g["sigma"]
                    else:
                        weights = np.sqrt(np.maximum(evals, 1e-16))
                        z = evecs @ (rng.normal(size=dimension) * weights)
                        z /= np.linalg.norm(z) + 1e-12
                        tangent = evecs[:, -1]
                        sign = 1 if rng.random() < 0.5 else -1
                        step = 0.35 * (local_best - x) + 0.65 * g["sigma"] * z + 0.25 * g["sigma"] * sign * tangent
                    Y.append(np.clip(x + step, lo, hi))

                if used + len(Y) > budget:
                    break
                Y = np.asarray(Y)
                yvals = np.array([evaluate(y) for y in Y], dtype=float)
                imp = yvals < vals
                success = float(np.mean(imp))
                X = np.where(imp[:, None], Y, X)
                vals = np.where(imp, yvals, vals)
                target_success = 0.22 if mode != "valley" else 0.28
                sigma = np.clip(g["sigma"] * np.exp(0.9 * (success - target_success)), 1e-6 * span, 0.30 * span)
                stall = 0 if np.any(imp) else g["stall"] + 1

                if stall >= 10 and mode == "multi":
                    worst = np.argsort(vals)[-max(1, per // 4):]
                    for idx in worst:
                        if used >= budget: break
                        X[idx] = rng.uniform(lo, hi, dimension)
                        val = evaluate(X[idx])
                        if val is not None: vals[idx] = val
                    stall = 0

                groups[gi] = {"X": X, "vals": vals, "sigma": sigma, "stall": stall,
                              "mode": mode, "success_hist": (g["success_hist"] + [success])[-8:]}
                if used >= budget: break

            if gen % 10 == 0 and used < budget and best is not None:
                for g in groups:
                    if used >= budget: break
                    if rng.random() < 0.5:
                        w = int(np.argmax(g["vals"]))
                        g["X"][w] = best.copy()
                        val = evaluate(best)
                        if val is not None: g["vals"][w] = val

        if best is None:
            allX = np.vstack([g["X"] for g in groups]); allV = np.concatenate([g["vals"] for g in groups])
            j = int(np.argmin(allV)); best = allX[j].copy(); bestv = float(allV[j])

        if used < budget:
            allX = np.vstack([g["X"] for g in groups]); allV = np.concatenate([g["vals"] for g in groups])
            elite = allX[np.argsort(allV)[:max(6, dimension)]]
            C = np.cov(elite, rowvar=False) + np.eye(dimension) * (1e-12 * span ** 2)
            sigma = 0.03 * span; x = best.copy()
            while used < budget and sigma > 1e-9 * span:
                batch = min(max(4, 2 * dimension), budget - used)
                Y = rng.multivariate_normal(x, C * (sigma / span) ** 2, size=batch)
                Y = np.clip(Y, lo, hi)
                yvals = np.array([evaluate(y) for y in Y], dtype=float)
                j = int(np.argmin(yvals))
                if yvals[j] < bestv:
                    step = Y[j] - x; x = Y[j].copy(); best = x.copy(); bestv = float(yvals[j])
                    C = 0.85 * C + 0.15 * np.outer(step, step) + np.eye(dimension) * 1e-14
                    sigma *= 0.92
                else:
                    sigma *= 0.72

        return OptimizationResult(best, bestv, used, seed, "RO-V4-historical")
