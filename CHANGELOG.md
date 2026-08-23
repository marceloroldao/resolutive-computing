# Changelog

All notable changes to Resolutive Computing will be documented in this file.

The project follows semantic versioning for public releases once the v1 API is frozen.

## [Unreleased]

### Planned
- Full v1 release benchmark review across multiple dimensions.
- Clean package build/install verification across supported Python versions.
- Final API compatibility review before `1.0.0rc1`.

## [0.2.0-rc2] - 2026-08-23

### Fixed
- Corrected `CITATION.cff` metadata for GitHub/Zenodo citation parsing.
- Added explicit release version and release date to citation metadata.
- Removed the custom SPDX-like license identifier from `CITATION.cff`; the authoritative project licensing terms remain in the repository `LICENSE` file.
- Bumped the Python package version to `0.2.0rc2`.

### Status
- This is a metadata-corrected pre-release replacing `v0.2.0-rc1` for citation/archival purposes.
- No scientific benchmark claims were changed by this metadata-only release correction.

## [0.2.0-rc1] - 2026-08-23

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
- Product target for this release candidate is PC/server first. Embedded targets such as ESP32 are explicitly deferred to a later track.
- Public API compatibility is guarded by regression tests.
- Multi-resolution budget validation now accounts for the minimum budget required by its Hybrid core.

### Validation status
- Monolithic and sequential ask/tell V2, Hybrid, and Multi-Resolution implementations matched in the repository's validated equivalence campaign.
- Checkpoint, registry, persistence, HTTP API, Python client, and concurrency test suites passed prior CI runs.
- This is a pre-release. The full v1 release benchmark and clean package build/install verification remain gates before `1.0.0rc1`.

## [0.1.0.dev0]

Development baseline preceding the v1 API and productization track.
