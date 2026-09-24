# Resolutive Computing API v1 Contract

Status: **release-candidate contract** for the PC/server v1.0 line.

This document defines the public behavior that must remain backward-compatible for the 1.x series unless a documented deprecation path is provided.

## Scope

The v1 API is a stateful black-box optimization service. The server proposes candidate points with `ask`; the caller evaluates those points externally and returns scalar objective values with `tell`.

The optimization engine never evaluates the user's objective function itself in remote-session mode.

## Session protocol

A session follows this state machine:

```text
create -> ask -> tell -> ask -> tell -> ... -> result
```

Only one batch may be pending for a session at a time. Calling `ask` again before the matching `tell` is a protocol conflict.

A persisted session must preserve a pending batch across process restart.

## HTTP endpoints

### `GET /health`

Returns HTTP 200:

```json
{"status":"ok"}
```

### `POST /sessions`

Creates a session.

Request:

```json
{
  "dimension": 10,
  "bounds": [-5.12, 5.12],
  "budget": 3000,
  "seed": 0,
  "mode": "multires",
  "batch_size": 16
}
```

Required invariants:

- `dimension >= 1`
- `budget >= 1`
- `batch_size >= 1`
- `bounds[0] < bounds[1]`
- supported modes: `prototype`, `hybrid`, `multires`

Response: HTTP 201 with session metadata.

### `GET /sessions/{session_id}`

Returns session metadata:

```json
{
  "session_id": "...",
  "done": false,
  "evaluations": 0,
  "remaining": 3000,
  "session_type": "HybridMultiResolutionSequentialSession"
}
```

### `POST /sessions/{session_id}/ask`

Returns the next candidate batch:

```json
{
  "generation": 0,
  "points": [[0.1, -0.2], [0.3, 0.4]]
}
```

A second `ask` while a batch is pending returns HTTP 409.

### `POST /sessions/{session_id}/tell`

Request:

```json
{"values":[1.25, 0.91]}
```

There must be exactly one finite scalar value for each point in the pending batch.

Response: HTTP 200 with updated session metadata.

Calling `tell` without a pending batch returns HTTP 409.

### `GET /sessions/{session_id}/result`

Returns the best known result once at least one observation exists:

```json
{
  "x": [0.0, 0.0],
  "fun": 0.0,
  "evaluations": 3000,
  "seed": 0,
  "optimizer": "...",
  "status": "success",
  "diagnostics": {}
}
```

If no observation exists yet, the server returns HTTP 409.

### `GET /sessions/{session_id}/checkpoint`

Returns a versioned JSON-safe checkpoint string. The checkpoint format is implementation-owned but must remain restorable by compatible 1.x versions according to the checkpoint compatibility policy.

### `DELETE /sessions/{session_id}`

Deletes the session. Persistent registries must also remove the on-disk checkpoint.

Response: HTTP 204.

## HTTP error contract

- `400` — invalid request or invalid `tell` values
- `404` — unknown session ID
- `409` — valid request shape but invalid session protocol state
- `422` — request schema validation failure produced by FastAPI/Pydantic

The Python client maps server errors to `ResolutiveHTTPError` and preserves both `status_code` and `detail`.

## Python client contract

Public classes:

- `ResolutiveClient`
- `RemoteSession`
- `RemoteAskBatch`
- `RemoteSessionInfo`
- `RemoteResult`
- `ResolutiveHTTPError`

Primary usage:

```python
from resolutive.client import ResolutiveClient

with ResolutiveClient("http://127.0.0.1:8000") as client:
    session = client.create(
        dimension=10,
        bounds=(-5.12, 5.12),
        budget=3000,
        seed=0,
        mode="multires",
    )

    while not session.info().done:
        batch = session.ask()
        values = [objective(x) for x in batch.points]
        session.tell(values)

    result = session.result()
```

`attach(session_id)` must reconnect to an existing server-side session without altering its state.

## Persistence requirements

For the v1 PC/server line, a persistent registry must guarantee:

1. atomic replacement of the current checkpoint file;
2. restoration of all valid sessions after process restart;
3. preservation of NumPy RNG state;
4. preservation of a pending `ask` batch;
5. deletion durability;
6. no use of `pickle` for persisted session data.

## Determinism and equivalence

For fixed optimizer configuration, seed, objective responses, and evaluation budget, checkpoint/restart must not change the optimization trajectory.

The incremental `ask/tell` implementations used by the v1 service are regression-tested against their monolithic counterparts for the repository's reference comparison cases.

## Compatibility policy

Within 1.x:

- existing endpoint paths and successful-response fields documented here must not be removed;
- new response fields may be added;
- new optional request fields may be added with backward-compatible defaults;
- semantic changes to `ask/tell` ordering require a major version;
- checkpoint format changes require explicit version migration or rejection with a clear compatibility error.

## Non-goals for v1.0

The first stable release targets PC/server deployments. Microcontroller/ESP32 constraints, embedded transports, distributed clustering, authentication/authorization, and multi-tenant policy are intentionally outside this v1 core contract unless separately documented.
