# Changelog

All notable changes to Resolutive Computing will be documented in this file.

The project follows semantic versioning for public releases once the v1 API is frozen.

## [Unreleased]

### Added
- Unified Python `optimize()` API.
- Stateful `ask()` / `tell()` optimization sessions.
- Sequential V2, Hybrid, and Multi-Resolution sessions preserving monolithic optimizer behavior in the validated comparison suite.
- JSON-safe versioned checkpoints without pickle.
- `SessionRegistry` and disk-backed `PersistentSessionRegistry`.
- FastAPI HTTP server for session lifecycle management.
- Official Python HTTP client.
- Concurrency and restart/recovery tests for PC/server deployments.
- V1 HTTP API contract and release checklist.
- Release benchmark harness against CMA-ES, SciPy Differential Evolution, Nelder-Mead, and Powell.

### Changed
- V1 development target is PC/server first. Embedded targets such as ESP32 are explicitly deferred to a later track.
- Public API compatibility is now guarded by regression tests.

### Validation status
- Monolithic and sequential ask/tell V2, Hybrid, and Multi-Resolution implementations matched in the repository's validated equivalence campaign.
- Checkpoint, registry, persistence, HTTP API, Python client, and concurrency test suites have passed prior CI runs.
- The full v1 release benchmark and clean package build/install verification remain release gates before `1.0.0rc1`.

## [0.1.0.dev0]

Development baseline preceding the v1 API and productization track.
