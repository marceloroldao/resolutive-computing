# Resolutive Computing v0.2.0-rc3

This pre-release supersedes `v0.2.0-rc1` and `v0.2.0-rc2` for citation and archival purposes.

## Why rc3 exists

The previous release tags were created with citation metadata that GitHub could not reliably parse. `v0.2.0-rc3` uses a minimal `CITATION.cff` aligned with GitHub's documented Citation File Format example.

## Metadata corrections

- Minimal CFF 1.2.0 citation metadata.
- Explicit release version `0.2.0-rc3`.
- Explicit release date `2026-08-23`.
- Custom non-SPDX license identifier removed from `CITATION.cff`.
- Project licensing remains authoritative in the repository `LICENSE` file.
- Python package version updated to `0.2.0rc3`.

## Software scope

The software scope is unchanged from the earlier v0.2.0 release candidates:

- Unified Python `optimize()` API.
- Stateful `ask()` / `tell()` optimization sessions.
- V2, Hybrid and Hybrid Multi-Resolution engines.
- JSON-safe checkpointing without pickle.
- In-memory and disk-backed session registries.
- FastAPI HTTP server.
- Official Python HTTP client.
- Restart/recovery semantics.
- Concurrent multi-session execution for PC/server use.
- Benchmark infrastructure against CMA-ES, Differential Evolution, Nelder-Mead and Powell.

## Validation status

Prior CI campaigns validated monolithic vs. stateful `ask/tell` equivalence for the tested V2, Hybrid and Multi-Resolution configurations. The full v1 multi-dimensional release benchmark and final packaging verification remain gates before the v1 API is declared stable.

## Scope

This release targets PC/server environments with Python 3.10+.

Embedded targets such as ESP32 are intentionally deferred to a later development track.

## Status

`v0.2.0-rc3` is a pre-release intended for evaluation, reproducibility, citation validation and API testing.
