# Resolutive Computing v1.0 Release Checklist

This checklist defines the release gates for the first PC/server-focused v1.0. A release candidate must not be promoted to stable v1.0 while any blocking gate remains open.

## Scope freeze

- [x] Primary target is PC/server execution.
- [x] ESP32/embedded optimization is explicitly deferred to a later release line.
- [x] Core black-box optimization remains independent from HTTP/server dependencies.
- [x] Optional extras remain separated: `client`, `server`, `bench`, `test`.

## Core algorithm

- [x] Monolithic and ask/tell specialist sessions are regression-tested for equivalence.
- [x] Hybrid ask/tell is regression-tested against the monolithic implementation.
- [x] MultiResolution ask/tell is regression-tested against the monolithic implementation.
- [x] Checkpoint/restore preserves the optimization trajectory and RNG state.
- [x] A pending `ask()` survives checkpoint/restore without generating a duplicate batch.

## Session runtime

- [x] `SessionRegistry` supports isolated concurrent sessions.
- [x] `PersistentSessionRegistry` restores sessions after process restart.
- [x] Persistent writes use atomic replacement rather than pickle-based object storage.
- [x] Same-session concurrent `ask()` conflicts are rejected.
- [x] Multi-session concurrency/stress tests pass.

## HTTP and Python client

- [x] HTTP lifecycle covers create/info/ask/tell/result/checkpoint/delete.
- [x] HTTP errors distinguish not-found and ask/tell protocol conflicts.
- [x] Python client supports create, attach, ask, tell, result, checkpoint and delete.
- [x] API v1 contract is documented in `API_V1_CONTRACT.md`.
- [ ] Final compatibility review confirms no accidental breaking changes to the documented v1 surface.

## Release benchmark — BLOCKING

- [x] Release benchmark script exists.
- [x] Strong external references include CMA-ES and SciPy optimizers.
- [x] Strict objective-call budgets are enforced.
- [x] Initial-point bias at known optima has been removed.
- [x] Raw and aggregated CSV output is produced by the campaign.
- [x] Automatic win/tie/loss analysis is available.
- [ ] Full release campaign completes for the frozen commit.
- [ ] Results are reviewed for failures, pathological regimes and unsupported superiority claims.
- [ ] Release notes state where Resolutive wins, ties and loses rather than claiming universal superiority.

## Packaging — BLOCKING

The `Package Validation` workflow automates the clean build/install gates below. Items remain unchecked until the frozen release commit completes that workflow successfully.

- [ ] Package version is changed from development metadata only when `v1.0-rc1` is actually cut.
- [ ] Source distribution and wheel build successfully from a clean checkout.
- [ ] Wheel installs in a clean Python 3.10 environment.
- [ ] Wheel installs in a clean Python 3.11 environment.
- [ ] Wheel installs in a clean Python 3.12 environment.
- [ ] `import resolutive` succeeds from the installed wheel.
- [ ] Minimal `optimize()` smoke succeeds from the installed wheel.
- [ ] Client extra installs and imports `ResolutiveClient`.
- [ ] Server extra installs and creates the FastAPI application.

## Public API freeze

The following top-level names are considered part of the v1 Python compatibility surface once `v1.0-rc1` is cut:

- `optimize`
- `create_session`
- `checkpoint_session`
- `restore_session`
- `checkpoint_json`
- `restore_json`
- `SessionRegistry`
- `PersistentSessionRegistry`
- `OptimizationResult`
- `OptimizationSession`
- `MultiResolutionSession`
- `MultiResolutionState`
- `AskBatch`
- `ResolutiveV2`
- `ResolutiveV5`

Additional experimental implementation classes may remain importable from submodules but are not automatically covered by the v1 compatibility promise.

The documented HTTP routes and Python client symbols are regression-guarded by `tests/test_api_v1_contract.py`.

## Documentation and release hygiene

- [x] README documents local Python, ask/tell, server and client usage.
- [x] Source-available/non-commercial licensing language is explicit.
- [ ] Changelog entry for `v1.0-rc1` is prepared from the frozen commit.
- [x] Known limitations are documented in `KNOWN_LIMITATIONS.md`.
- [ ] Benchmark configuration and exact frozen commit are cited in release notes.
- [ ] `CITATION.cff` version/date are synchronized with the release candidate.
- [ ] Tag and GitHub release are created only after all blocking RC gates pass.

## Promotion rule

`v1.0-rc1` may be cut when the full release benchmark, packaging checks and API compatibility review are green. Stable `v1.0` requires the RC to remain reproducible with no release-blocking defects discovered during final validation.
