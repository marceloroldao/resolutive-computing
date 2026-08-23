# Resolutive Computing v0.2.0-rc2

This pre-release supersedes `v0.2.0-rc1` for citation and archival purposes.

## Why rc2 exists

`v0.2.0-rc1` was published with citation metadata that could not be reliably parsed by GitHub/Zenodo. The software and benchmark content of the release candidate remain unchanged in scope; `rc2` corrects the release metadata and package versioning.

## Metadata fixes

- `CITATION.cff` now declares `version: "0.2.0-rc2"`.
- `CITATION.cff` now declares `date-released: "2026-08-23"`.
- The custom `LicenseRef-RRNCL-1.0` identifier was removed from `CITATION.cff` to improve citation parser compatibility.
- The authoritative Resolutive Research and Non-Commercial License remains in the repository `LICENSE` file.
- Python package version is now `0.2.0rc2`.

## Included functionality

The release candidate continues to include:

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

## Validation status

Prior CI campaigns validated exact monolithic-vs-ask/tell equivalence for the tested V2, Hybrid and Multi-Resolution scenarios. The repository also contains controlled 10D comparisons and strong-baseline experiments.

Representative prior 10D results showed `RO-Hybrid-Multires-exp` competitive with strong external baselines: better median results than CMA-ES on the tested Rastrigin and Rosenbrock campaigns, while CMA-ES remained stronger on Ackley. These are benchmark-specific experimental results, not a claim of universal superiority.

## Scope

This release candidate targets desktop and server environments with Python 3.10+.

Embedded targets, including ESP32, are intentionally outside the scope of this release and will be addressed in a later track.

## Status

`v0.2.0-rc2` is a pre-release intended for evaluation, reproducibility, API feedback and archival/citation testing before the v1 API is declared stable.

The remaining v1 gates include the full multi-dimensional release benchmark, clean package build/install verification and final public API compatibility review.

## License

Resolutive Computing is distributed under the Resolutive Research and Non-Commercial License (RRNCL) v1.0. Academic, research, educational and other non-commercial uses are permitted under the license terms. Commercial use requires separate written authorization.
