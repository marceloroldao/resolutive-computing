# Resolutive Computing

**Resolutive Computing** is a modular research and engineering framework for resolutive computation, black-box optimization, adaptive search, benchmarking, and scientific applications.

The project begins with the Resolutive Optimizer family (RO), developed as a reproducible optimization research program. The repository is designed to evolve toward scientific computing domains where adaptive, memory-aware, geometry-sensitive search may be useful.

## Status

Experimental research framework. Results must be reproduced from versioned repository code before being treated as scientific evidence.

**RSMS compatibility:** `1.0-rc.1`  
See [`RSMS_COMPATIBILITY.md`](RSMS_COMPATIBILITY.md).

## Initial scope

- Resolutive Optimizer implementations (RO-V2 and RO-V5)
- Continuous black-box benchmark functions
- Budget-matched classical baselines
- Reproducible multi-seed experiments
- Controlled RO-V5 ablation studies
- Dimensional scaling tests
- Future COCO/BBOB integration

## Scientific principles

1. Same evaluation budget for competing optimizers.
2. Multiple random seeds.
3. Median, mean, variance, best and worst-case reporting.
4. Ablation before attributing gains to a mechanism.
5. No benchmark claim without reproducible code and recorded configuration.
6. Keep experimental results separate from theoretical interpretation.
7. Preserve favorable, neutral, and negative results under the same reporting standard.

## Quick start

```bash
python -m pip install -e ".[test,bench]"
python -m pytest -q
```

Run the classical 10-dimensional benchmark suite:

```bash
python benchmarks/run_classical.py --dimension 10 --budget 6000 --seeds 16
```

Run a dimensional scaling experiment:

```bash
python benchmarks/run_scaling.py --dimensions 10 20 30 50 --budget-per-dimension 600 --seeds 8
```

Run a controlled RO-V5 ablation study:

```bash
python benchmarks/run_ablation.py --dimension 10 --budget 6000 --seeds 16
```

Generated quantitative results should be stored under `results/` and associated with the exact commit, configuration, and seed set used to produce them.

## Current reference methods

- `RandomSearch` — exact-budget uniform random baseline.
- `SimulatedAnnealing` — exact-budget stochastic local/global baseline.
- `DifferentialEvolution` — exact-budget `DE/rand/1/bin` baseline.
- `RO-V2` — single-funnel resolutive baseline with coherence-controlled exploration and coordinate collapse.
- `RO-V5` — hybrid resolutive engine with stagnation-triggered multi-funnel escape.

The classical baselines are reference implementations under the same objective-call budget. They are not yet a complete state-of-the-art benchmark panel; COCO/BBOB and additional established optimizers remain part of the validation roadmap.

## Author

Marcelo Roldão Matos  
ORCID: 0009-0003-6075-4680

## License

Copyright (c) 2026 Marcelo Roldão Matos.

Resolutive Computing is **source-available for academic, educational, scientific research, evaluation, and other qualifying non-commercial use** under the terms in [`LICENSE`](LICENSE).

Commercial use is not granted by the public license and requires prior written authorization under a separate commercial agreement. See [`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md).

Because commercial use is restricted, this project should **not** be described as OSI-approved open source.
