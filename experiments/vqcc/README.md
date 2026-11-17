# VQCC inverse-square-root experiment

An isolated experiment for Google Quantum AI's Verifiable Quantum Circuit
Compilation competition. It changes no optimizer/core code and submits nothing
to Kaggle.

## What is actually tested

- A conventional quadratic approximation over four subintervals of [1,2),
  rescaled to [1,4), followed by one Newton step.
- A real native kickmix circuit using the reference normalization, exact
  column-truncated arithmetic and measurement-based uncomputation.
- Full-domain integer screening for three precision configurations.
- A negative control: the verifier must reject a circuit with no target phase.
- Full native verification of both the reference and the selected candidate:
  3,133,452 inputs times 64 shots = 200,540,928 shots per circuit.
- Measured Q, Toffoli-equivalent C (including rotation synthesis), and Q*C.

The screening model reproduces column truncation, which differs from flooring a
complete mathematical multiplication. The earlier conversation's generic fixed
point results therefore are not used as circuit acceptance evidence.

Configuration selection uses Q times static Toffoli count only as a cheap proxy.
The final comparison always uses the verifier's full cost including rotations.
A worse result is retained. A passing workflow does not imply lower cost.

This experiment does not test RO superiority, establish new physics, or estimate
a probability of winning. Even beating the initial reference would not establish
competitiveness against the current leaderboard.

## Source provenance

The reference generator and competition verifier are downloaded unchanged at
runtime from two public mirrors and their Git blob IDs must agree:

- Inzainos/quantum-circuit-compilation at
  4277c767440df9efb8a90860ae2ce47028e46ba9
- nhphan0505/verifiable-quantum-circuit at
  872bb089cbf62931fd156f5c86243c7bd053c9a0

Expected blobs:
- nr_reference_circuit.py: 4fb63a258e1eea2f35312f77785902f9bcc914d5
- verify_submission_invsqrt.py: 1fa529e9790a5cb000ee7f084e197c9252e5f9d7

These are public copies of the competition package, not an authenticated
Kaggle download. Matching mirrors and the paper contract are useful provenance
checks, not proof of authenticity. Before an actual competition submission,
compare these files with a fresh official Kaggle download.

The upstream package's Apache-2.0 license is retained in the results artifact.
No third-party code is copied into this repository. Google kickmix is built
from quantumlib/Qualtran at f559b12af459e7e97e6c6c382434431c03b903c9.
The scientific contract is described in https://arxiv.org/abs/2610.23832.

## Run

The GitHub Actions workflow VQCC invsqrt experiment runs on pushes to the
experiment/vqcc-invsqrt branch. No secrets, Kaggle account or VM are required.
Its artifact contains the generated circuits, full verifier JSON, stdout,
stderr, source hashes, integer-screening outcomes and SUMMARY.md.

Local use after installing numpy and native kickmix:

    python experiments/vqcc/run.py

The complete native run can take several minutes. The runner has explicit
timeouts and fails if a purported full PASS checks fewer than 200,540,928 shots.
