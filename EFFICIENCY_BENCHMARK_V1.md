# V1 Computational Efficiency Benchmark — Findings

This document records the first methodologically isolated computational-efficiency campaign for the PC/server v1 line.

## Campaign

- Commit: `3b9504742d89ce4af78688fb54601bd4c7dcf445`
- GitHub Actions run: `32837996828`
- Runner: GitHub-hosted Ubuntu 24.04, x86_64
- Python: 3.12.14
- Dimensions: 2, 10, 30
- Seeds: 10
- Benchmarks: Ackley, Rastrigin, Rosenbrock
- Methods: RO-Hybrid-Multires, CMA-ES (pycma), Differential Evolution (SciPy), Nelder-Mead (SciPy), Powell (SciPy)
- Objective-call budget: `max(1000, 300 * dimension)`
- Total runs: 450
- Aggregated scenarios: 45

Each benchmark/method/dimension/seed execution ran in a fresh Python subprocess so process-retained memory from one optimizer could not contaminate another. Optimizer order was rotated deterministically by seed. Wall-clock time and RSS remain runner-sensitive and must only be interpreted comparatively within a campaign.

## Main result

The current RO-Hybrid-Multires implementation is **not computationally dominant** over the reference optimizers in this benchmark family. In particular, Powell is substantially more efficient on these smooth analytic functions and often reaches better final objective values with far fewer objective evaluations, lower wall/CPU time and lower incremental RSS.

This is a valid negative result and blocks any claim of universal or general computational superiority for the current optimizer.

## Representative comparisons

### Ackley, 30D

| Method | Median final objective | Median evaluations | Median evals to 1e-6 | Median wall time | Median RSS delta |
|---|---:|---:|---:|---:|---:|
| Powell | 2.13e-14 | 5,536 | 256 | 0.136 s | 159,744 B |
| CMA-ES | 9.28e-10 | 9,000 | 6,539.5 | 1.668 s | 4,177,920 B |
| RO-Hybrid-Multires | 4.38e-08 | 8,951 | 8,492 | 1.509 s | 4,311,040 B |
| DE | 12.34 | 9,000 | not reached | 0.840 s | 831,488 B |
| Nelder-Mead | 19.68 | 9,000 | not reached | 0.442 s | 491,520 B |

### Rastrigin, 30D

| Method | Median final objective | Median evaluations | Median evals to 1e-6 | Median wall time | Median RSS delta |
|---|---:|---:|---:|---:|---:|
| Powell | 0.0 | 2,735.5 | 281 | 0.052 s | 159,744 B |
| RO-Hybrid-Multires | 16.42 | 8,968 | not reached | 1.406 s | 4,313,088 B |
| CMA-ES | 52.24 | 9,000 | not reached | 1.587 s | 4,173,824 B |
| DE | 260.88 | 9,000 | not reached | 0.728 s | 831,488 B |
| Nelder-Mead | 291.40 | 9,000 | not reached | 0.351 s | 491,520 B |

On Rastrigin-30D, RO-Hybrid-Multires produces a materially better median final objective than CMA-ES, DE and Nelder-Mead, but still loses decisively to Powell in this campaign.

### Rosenbrock, 30D

| Method | Median final objective | Median evaluations | Median wall time | Median RSS delta |
|---|---:|---:|---:|---:|
| Powell | 0.662 | 9,000 | 0.193 s | 159,744 B |
| RO-Hybrid-Multires | 1.939 | 8,943 | 1.000 s | 4,298,752 B |
| CMA-ES | 21.48 | 9,000 | 1.531 s | 4,163,584 B |
| DE | 13,074.37 | 9,000 | 0.701 s | 831,488 B |
| Nelder-Mead | 38,047.26 | 9,000 | 0.357 s | 491,520 B |

RO-Hybrid-Multires is clearly stronger than CMA-ES, DE and Nelder-Mead on median final objective here, but Powell remains stronger and much cheaper.

## Engineering interpretation

The present bottleneck is not basic optimizer viability. The principal weakness is **evaluation efficiency**: the multiresolution/hybrid search spends too much of the objective-call budget exploring before entering a productive local-convergence regime.

The current implementation also carries a relatively large incremental process-memory footprint (~4.2 MB median in many scenarios) compared with SciPy local optimizers (~0.16–0.50 MB). This is small in absolute PC/server terms, but it is not an efficiency advantage.

CPU-time behavior also shows that the Resolutive implementation currently performs significant Python-side search-management work. It cannot yet support a claim that fewer external objective calls are being exchanged for cheaper internal computation; in several important cases it uses both more objective calls and more compute than Powell.

## What is supported by the evidence

The campaign supports these narrower statements:

1. RO-Hybrid-Multires is a viable black-box optimizer across the tested families.
2. It can outperform Differential Evolution and Nelder-Mead on solution quality in difficult higher-dimensional regimes.
3. It can outperform CMA-ES on median final objective in some multimodal/high-dimensional cases, notably Rastrigin-30D and Rosenbrock-30D in this campaign.
4. It is not universally superior and is currently dominated by Powell on the tested analytic suite.
5. The next optimization target should be early regime detection and aggressive transition from global/multiresolution exploration to local refinement.

## Next technical gate

Before claiming an efficiency advantage, a new candidate should demonstrate at least one of the following under the same strict objective-call protocol:

- substantially fewer evaluations to target than the current RO-Hybrid-Multires without degrading robustness;
- a Pareto advantage against Powell/CMA-ES on a problem class where derivative-free local structure alone is insufficient;
- materially lower internal CPU/RSS cost while preserving the existing solution-quality advantage over DE/Nelder-Mead/CMA-ES in selected regimes.

A universal-win requirement is not appropriate. The intended release documentation should identify the problem regimes where Resolutive is advantageous and the regimes where established local optimizers are preferable.
