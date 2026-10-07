# Resolutive Computing

**Resolutive Computing** is a modular research and engineering framework for resolutive computation, black-box optimization, adaptive search, benchmarking, and PC/server deployment.

The current v1.0 development line focuses on continuous black-box optimization with deterministic stateful `ask/tell` sessions, checkpoint/restart, multi-session registries, an optional HTTP service, and a Python client.

## Status

**Maturity:** pre-v1.0 / release-development.

Results must be reproduced from code before being treated as scientific or engineering evidence. Computational performance does not validate any underlying physical interpretation.

The first stable release targets **PCs and servers**. Embedded/ESP32 work is intentionally deferred to a later development line.

## What the project does

For a black-box objective

```text
f(x) -> scalar cost
```

Resolutive Computing can either optimize directly in-process or operate as a stateful external-evaluation engine:

```text
ask -> external evaluation -> tell -> ask -> ...
```

This makes the optimizer usable with simulations, engineering software, remote processes, expensive evaluations, and other systems where the objective function should remain outside the optimizer process.

## Core capabilities on the v1 line

- RO-V2 and RO-V5 optimizer family
- Hybrid regime routing
- Hybrid multi-resolution optimization
- Sequential `ask/tell` sessions regression-tested against monolithic engines
- Versioned JSON-safe checkpoints without `pickle`
- Checkpoint/restart trajectory preservation
- In-memory `SessionRegistry`
- Atomic disk-backed `PersistentSessionRegistry`
- Concurrent independent sessions on PC/server
- Optional FastAPI HTTP transport
- Optional synchronous Python HTTP client
- Reproducible benchmark, scaling, generalization, PID and MPPT campaigns

## Installation

Core library:

```bash
pip install -e .
```

HTTP server dependencies:

```bash
pip install -e ".[server]"
```

Python client dependencies:

```bash
pip install -e ".[client]"
```

Development/test environment:

```bash
pip install -e ".[test,bench]"
```

## Direct Python optimization

```python
import numpy as np
from resolutive import optimize


def sphere(x):
    x = np.asarray(x, dtype=float)
    return float(np.dot(x, x))


result = optimize(
    sphere,
    dimension=10,
    bounds=(-5.0, 5.0),
    budget=3000,
    seed=0,
)

print(result.fun)
print(result.x)
```

## Stateful local `ask/tell`

```python
import numpy as np
from resolutive import create_session

session = create_session(
    dimension=10,
    bounds=(-5.12, 5.12),
    budget=3000,
    seed=0,
    mode="multires",
)

while not session.done:
    batch = session.ask()
    values = np.sum(batch.points * batch.points, axis=1)
    session.tell(values)

print(session.result())
```

Only one batch may be pending per session. `tell()` must complete the current batch before the next `ask()`.

## Checkpoint and restart

```python
from resolutive import checkpoint_json, restore_json

payload = checkpoint_json(session)
restored = restore_json(payload)
```

The checkpoint preserves optimizer state, NumPy RNG state, multi-resolution phase, and a pending `ask` batch. The format is JSON-safe and deliberately avoids `pickle`.

## Session registries

For multiple independent sessions in one process:

```python
from resolutive import SessionRegistry

registry = SessionRegistry()
sid = registry.create(
    dimension=10,
    bounds=(-5.0, 5.0),
    budget=3000,
    mode="multires",
)
```

For restart-safe PC/server operation:

```python
from resolutive import PersistentSessionRegistry

registry = PersistentSessionRegistry("./resolutive-state")
```

The persistent registry writes each session atomically using `fsync` plus `os.replace` and restores valid sessions automatically when the process starts again.

## HTTP API

Install the server extra and run the FastAPI factory:

```bash
uvicorn resolutive.http_api:create_app --factory --host 127.0.0.1 --port 8000
```

Core endpoints:

```text
GET    /health
POST   /sessions
GET    /sessions/{id}
POST   /sessions/{id}/ask
POST   /sessions/{id}/tell
GET    /sessions/{id}/result
GET    /sessions/{id}/checkpoint
DELETE /sessions/{id}
```

The release-candidate compatibility contract is documented in [`API_V1_CONTRACT.md`](API_V1_CONTRACT.md).

## Python HTTP client

```python
import numpy as np
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
        values = np.sum(batch.points * batch.points, axis=1)
        session.tell(values)

    result = session.result()
    print(result.fun)
```

A client can reconnect to a pre-existing server-side session with:

```python
session = client.attach(session_id)
```

## Reliability properties currently tested

The CI validates Python 3.10, 3.11 and 3.12. The current test line includes:

- monolithic vs incremental specialist equivalence;
- monolithic vs incremental Hybrid equivalence;
- monolithic vs incremental Multires equivalence;
- checkpoint/restart trajectory preservation;
- restart with a pending `ask` batch;
- multiple independent persisted sessions;
- concurrent independent server sessions;
- conflicting concurrent `ask` calls on one session;
- classical benchmarks, dimensional scaling and generalization;
- PID and MPPT application smoke tests;
- strong external optimization baselines.

## Relationship to Resolutive Science

`resolutive-science` is the normative source of truth for shared Resolutive Science terminology, notation and scientific-status conventions.

- Resolutive Science repository baseline: `v0.1.1`
- RSMS compatibility: `1.0-rc.1` — candidate compatibility, subject to re-audit when RSMS 1.0 becomes stable
- Project governance baseline: `RSPS 1.0-draft`

Resolutive Computing may use computational constructs inspired by Resolutive Science, but such constructs are engineering hypotheses unless separately established by scientific evidence. Any project-specific semantic deviation from RSMS must be documented rather than silently redefining a shared term.

## Scientific principles

1. Same evaluation budget for competing optimizers.
2. Multiple random seeds.
3. Median, mean, variance, best and worst-case reporting.
4. Ablation before attributing gains to a mechanism.
5. No benchmark claim without reproducible code and recorded configuration.
6. Keep experimental results separate from theoretical interpretation.
7. Preserve scientifically relevant negative or inconclusive results.

## v1.0 release path

The remaining release work is intentionally focused on PC/server productization:

- freeze and regression-test the public API contract;
- complete the final release benchmark matrix against strong baselines;
- verify clean-package installation and server/client startup;
- finalize release documentation and changelog;
- cut a release candidate;
- run the full reproducibility campaign from the release candidate;
- tag v1.0 only after the release candidate remains green.

## Licensing

This repository is source-available for academic, educational and non-commercial research use under `LICENSE`.

Commercial use requires separate written authorization or a commercial license from the rights holder. Because commercial use is restricted, this project must not be described as OSI-approved open-source software.

## Citation

Citation metadata is provided in `CITATION.cff`. Cite the exact release or commit used until an archival DOI is assigned.

## Author

Marcelo Roldão Matos  
ORCID: 0009-0003-6075-4680
