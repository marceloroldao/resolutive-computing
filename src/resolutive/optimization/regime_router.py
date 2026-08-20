"""Experimental explicit regime router for black-box optimization.

The router spends a small diagnostic budget to estimate observable landscape
properties, converts them into transparent regime scores, and dispatches the
remaining budget to an existing optimizer. It intentionally uses no neural
network: all features and routing scores are inspectable and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .common import Objective, OptimizationResult, validate_bounds
from .hybrid_multires import ResolutiveHybridMultiResolution
from .hybrid_multires_robust import ResolutiveHybridMultiResolutionRobust
from .v5 import ResolutiveV5
from .v6 import ResolutiveV6


@dataclass(frozen=True)
class RegimeDiagnostics:
    noise_ratio: float
    ruggedness: float
    anisotropy: float
    curvature_proxy: float
    scores: dict[str, float]
    selected: str


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    keys = list(scores)
    vals = np.asarray([scores[k] for k in keys], dtype=float)
    vals -= np.max(vals)
    exp = np.exp(vals)
    probs = exp / np.sum(exp)
    return {k: float(p) for k, p in zip(keys, probs)}


class ResolutiveRegimeRouter:
    """Route to V5, V6, Multires, or Robust from explicit diagnostics."""

    def __init__(self, probe_fraction: float = 0.06, repeats: int = 3) -> None:
        if not 0.03 <= probe_fraction <= 0.15:
            raise ValueError("probe_fraction must be in [0.03, 0.15]")
        if repeats < 3 or repeats % 2 == 0:
            raise ValueError("repeats must be an odd integer >= 3")
        self.probe_fraction = float(probe_fraction)
        self.repeats = int(repeats)
        self.last_diagnostics: RegimeDiagnostics | None = None

    def _diagnose(self, objective: Objective, dimension: int, lo: float, hi: float,
                  budget: int, seed: int) -> tuple[RegimeDiagnostics, int]:
        rng = np.random.default_rng(seed + 440_021)
        span = hi - lo
        probe_budget = max(48, int(round(budget * self.probe_fraction)))
        probe_budget = min(probe_budget, max(48, budget // 5))

        # Repeated center evaluation estimates stochastic inconsistency.
        center = np.full(dimension, (lo + hi) / 2.0)
        center_vals = [float(objective(center)) for _ in range(self.repeats)]
        used = self.repeats
        center_med = float(np.median(center_vals))
        noise_abs = float(np.median(np.abs(np.asarray(center_vals) - center_med)))

        n = max(12, min(40, probe_budget - used))
        points = rng.uniform(lo, hi, size=(n, dimension))
        values = np.asarray([float(objective(x)) for x in points], dtype=float)
        used += n

        scale = float(np.median(np.abs(values - np.median(values)))) + 1e-12
        noise_ratio = float(1.4826 * noise_abs / scale)

        # Ruggedness: nearby points that change objective disproportionately.
        dmat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        nn = np.argmin(dmat, axis=1)
        local_df = np.abs(values - values[nn])
        local_dx = np.maximum(dmat[np.arange(n), nn] / span, 1e-12)
        ruggedness = float(np.median(local_df / local_dx) / (scale + 1e-12))

        # Anisotropy proxy from elite covariance spectrum.
        elite_count = max(5, n // 4)
        elite = points[np.argsort(values)[:elite_count]]
        cov = np.cov(elite, rowvar=False) + np.eye(dimension) * 1e-12
        eig = np.linalg.eigvalsh(cov)
        anisotropy = float((eig[-1] + 1e-12) / (eig[0] + 1e-12))
        log_aniso = float(np.log10(max(anisotropy, 1.0)))

        # Curvature proxy: objective departure from a local affine trend.
        X = np.column_stack([np.ones(n), points])
        coef, *_ = np.linalg.lstsq(X, values, rcond=None)
        residual = values - X @ coef
        curvature_proxy = float(np.median(np.abs(residual)) / (scale + 1e-12))

        # Transparent log-scores. They are hypotheses to be validated by ablation,
        # not learned weights. V5 favors rugged multimodality; V6 anisotropic valleys;
        # Multires smooth low-noise refinement; Robust confirmed stochastic noise.
        raw_scores = {
            "V5": 1.20 * np.log1p(ruggedness) + 0.25 * curvature_proxy - 1.20 * noise_ratio,
            "V6": 0.95 * log_aniso + 0.35 * curvature_proxy - 0.60 * noise_ratio,
            "Multires": 1.10 - 0.65 * np.log1p(ruggedness) - 0.35 * log_aniso - 2.20 * noise_ratio,
            "Robust": 3.25 * noise_ratio + 0.15 * curvature_proxy - 0.25 * np.log1p(ruggedness),
        }
        probs = _softmax({k: float(v) for k, v in raw_scores.items()})
        selected = max(probs, key=probs.get)
        diag = RegimeDiagnostics(
            noise_ratio=noise_ratio,
            ruggedness=ruggedness,
            anisotropy=anisotropy,
            curvature_proxy=curvature_proxy,
            scores=probs,
            selected=selected,
        )
        return diag, used

    def minimize(self, objective: Objective, *, dimension: int,
                 bounds: tuple[float, float], budget: int = 6000,
                 seed: int = 0) -> OptimizationResult:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        lo, hi = validate_bounds(bounds)
        if budget < 1200:
            raise ValueError("budget must be >= 1200")

        diag, used = self._diagnose(objective, dimension, lo, hi, budget, seed)
        self.last_diagnostics = diag
        remaining = budget - used

        if diag.selected == "V5":
            optimizer = ResolutiveV5()
        elif diag.selected == "V6":
            optimizer = ResolutiveV6()
        elif diag.selected == "Robust":
            optimizer = ResolutiveHybridMultiResolutionRobust()
        else:
            optimizer = ResolutiveHybridMultiResolution()

        # Ensure selected optimizer receives a legal budget. The router itself is
        # experimental and reports total objective evaluations including probing.
        min_required = 1000 if diag.selected == "Robust" else 800 if diag.selected == "Multires" else 200
        if remaining < min_required:
            optimizer = ResolutiveV5()
            diag = RegimeDiagnostics(diag.noise_ratio, diag.ruggedness, diag.anisotropy,
                                     diag.curvature_proxy, diag.scores, "V5")
            self.last_diagnostics = diag

        result = optimizer.minimize(
            objective, dimension=dimension, bounds=(lo, hi),
            budget=remaining, seed=seed,
        )
        return OptimizationResult(
            result.x, result.fun, used + result.evaluations, seed,
            f"RO-RegimeRouter-exp[{diag.selected}]",
        )
