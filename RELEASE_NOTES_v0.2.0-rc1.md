# Resolutive Computing v0.2.0-rc1

This release candidate is the first PC/server-oriented public preview of the Resolutive Computing productization track.

## Highlights

- Unified Python `optimize()` entry point.
- Stateful `ask()` / `tell()` optimization sessions.
- V2, Hybrid and Hybrid Multi-Resolution engines available through the session API.
- JSON-safe checkpointing without pickle.
- In-memory and disk-backed session registries.
- FastAPI HTTP service for remote optimization sessions.
- Official Python HTTP client.
- Restart/recovery semantics, including pending `ask()` state.
- Multi-session concurrency tests for PC/server use.
- Release benchmark harness against CMA-ES, SciPy Differential Evolution, Nelder-Mead and Powell.

## Validation available in the repository

Prior CI campaigns validated exact monolithic-vs-ask/tell equivalence for the tested V2, Hybrid and Multi-Resolution scenarios. The repository also contains controlled 10D comparisons and strong-baseline experiments.

Representative prior 10D results showed `RO-Hybrid-Multires-exp` competitive with strong external baselines: notably better median results than CMA-ES on the tested Rastrigin and Rosenbrock campaigns, while CMA-ES remained stronger on Ackley. These are benchmark-specific experimental results, not a claim of universal superiority.

## Scope

This release candidate targets desktop and server environments with Python 3.10+.

Embedded targets, including ESP32, are intentionally outside the scope of this release and will be addressed in a later track.

## Status

`v0.2.0-rc1` is a pre-release. It is intended for evaluation, reproducibility, API feedback and continued benchmarking before the v1 API is declared stable.

The remaining v1 gates include the full multi-dimensional release benchmark, clean package build/install verification and final public API compatibility review.

## License

Resolutive Computing is distributed under the Resolutive Research and Non-Commercial License (RRNCL) v1.0. Academic, research, educational and other non-commercial uses are permitted under the license terms. Commercial use requires separate written authorization.
