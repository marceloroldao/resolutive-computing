# Known Limitations — PC/Server v1 Line

This document records known limitations for the first stable PC/server-oriented Resolutive Computing line. These constraints are deliberate release boundaries unless stated otherwise.

## Deployment scope

- The v1 line targets Python 3.10+ on PCs and servers.
- ESP32, microcontrollers, hard real-time firmware and embedded transports are outside the v1 scope.
- The HTTP service is currently a single-process application. Distributed clustering, leader election and cross-node session migration are not part of the v1 contract.

## Security and tenancy

- The built-in HTTP transport does not provide authentication, authorization, TLS termination, API keys, quotas or tenant isolation.
- Production deployments that expose the service beyond a trusted host or network must place it behind an appropriate security boundary or gateway.
- Multi-tenant policy is not part of the current core API contract.

## Optimization model

- The stable v1 contract is focused on bounded continuous black-box optimization with scalar objective values.
- Constraint handling, mixed discrete/continuous variables, categorical variables and multi-objective Pareto optimization are not yet stable public features.
- Remote `ask/tell` sessions assume the caller returns exactly one finite scalar objective value for every point in the pending batch.
- Only one `ask` batch may be pending per session.

## Performance claims

- No optimizer in this repository should be treated as universally superior.
- Reported wins, ties and losses are benchmark-, dimension-, budget- and seed-dependent.
- Benchmark results are engineering evidence only for the exact recorded configuration and commit.
- Performance comparisons must use equivalent objective-call budgets and reproducible configurations.

## Persistence and recovery

- `PersistentSessionRegistry` is intended for local restart recovery, not distributed consensus or replicated storage.
- Atomic file replacement protects individual checkpoint writes, but it does not provide database-style transactions across multiple sessions.
- Checkpoint compatibility is versioned and must follow the compatibility policy defined in `API_V1_CONTRACT.md`; incompatible formats may be rejected explicitly.

## HTTP client and transport

- The official Python HTTP client is synchronous.
- Automatic retries, backoff, service discovery, load balancing and circuit breaking are not currently part of the client contract.
- Network transport does not change optimizer determinism guarantees when the same ordered objective responses are supplied, but external evaluation systems remain responsible for their own reproducibility.

## Scientific interpretation

Resolutive Computing is an engineering and optimization framework. Computational benchmark performance does not by itself validate physical, cosmological or other scientific interpretations associated with the broader Resolutive Science research program.
