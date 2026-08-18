# Resolutive Computing

**Resolutive Computing** is a modular research and engineering framework for resolutive computation, black-box optimization, adaptive search, benchmarking, and scientific applications.

The project begins with the Resolutive Optimizer family (RO), developed as a reproducible optimization research program. The repository is designed to evolve toward scientific computing domains where adaptive, memory-aware, geometry-sensitive search may be useful.

## Status

**Maturity:** experimental / alpha research framework.

Results must be reproduced from code before being treated as scientific or engineering evidence. Computational performance does not validate any underlying physical interpretation.

## Relationship to Resolutive Science

`resolutive-science` is the normative source of truth for shared Resolutive Science terminology, notation and scientific-status conventions.

- Resolutive Science repository baseline: `v0.1.1`
- RSMS compatibility: `1.0-rc.1` — candidate compatibility, subject to re-audit when RSMS 1.0 becomes stable
- Project governance baseline: `RSPS 1.0-draft`

Resolutive Computing may use computational constructs inspired by Resolutive Science, but such constructs are engineering hypotheses unless separately established by scientific evidence. Any project-specific semantic deviation from RSMS must be documented rather than silently redefining a shared term.

## Initial scope

- Resolutive Optimizer implementations (RO-V2 and RO-V5)
- Continuous black-box benchmark functions
- Reproducible multi-seed experiments
- Ablation studies
- Dimensional scaling tests
- Future COCO/BBOB integration

## Scientific principles

1. Same evaluation budget for competing optimizers.
2. Multiple random seeds.
3. Median, mean, variance, best and worst-case reporting.
4. Ablation before attributing gains to a mechanism.
5. No benchmark claim without reproducible code and recorded configuration.
6. Keep experimental results separate from theoretical interpretation.
7. Preserve scientifically relevant negative or inconclusive results.

## Licensing

This repository is source-available for academic, educational and non-commercial research use under `LICENSE`.

Commercial use requires separate written authorization or a commercial license from the rights holder. Because commercial use is restricted, this project must not be described as OSI-approved open-source software.

## Citation

Citation metadata is provided in `CITATION.cff`. Cite the exact release or commit used until an archival DOI is assigned.

## Author

Marcelo Roldão Matos  
ORCID: 0009-0003-6075-4680
