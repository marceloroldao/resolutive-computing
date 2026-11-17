# Comparison with public VQCC work

Checked on 16 November 2026 (America/Sao_Paulo). This is a source comparison,
not a fresh rerun of a competitor's circuit and not a live leaderboard snapshot.

| Inverse-square-root circuit | Qubits | Task cost Q*C | Improvement over original reference | Evidence |
|---|---:|---:|---:|---|
| Original reference | 260 | 2,076,774.233715 | 1.0000x | Our complete native verification |
| Our first candidate | 241 | 1,728,155.547512 | 1.2017x | Our complete native verification, 200,540,928 shots |
| Public circuit af4698478fba, credited to Agniv Banerjee | 62 | 97,921.579543 | 21.2085x | Public receipt preserved by EigenOS, dated 5 November; not rerun by us |

Sources:
- [Our measured report](run-43990346433.md)
- [Public receipt](https://github.com/EigenOS-Quantum/eigenos-vqcc/blob/main/receipts/v7-invsqrt-verify.json)
- [Public portfolio and dated provenance](https://github.com/EigenOS-Quantum/eigenos-vqcc/blob/main/README.md)
- [Competition evaluation](https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation/overview/evaluation)

The public receipt identifies the full circuit SHA-256 as
af4698478fbad775a5a271690cf4c2d8783caf0b8d12048106d1973c4ac6862e,
and records a verifier-report hash. Its note says a fresh EigenOS full-domain
run was still in flight when that receipt was generated. We therefore treat
the cost as a published measurement, not independently reproduced evidence.

Our candidate's cost is **17.6484 times** that published cost. Even a dated
public comparator is substantially cheaper. This demonstrates that improving
the original baseline by 16.8% does not establish prize competitiveness.

Salts differ: our verification used resolutive-vqcc-experiment-v1; the public
receipt describes the default salt. This is not a controlled same-salt A/B
experiment. No precise current rank is inferred.

The live leaderboard page was not machine-readable in this check. Search
indexes returned dated/cached snapshots; those are not represented here as
the current standings.

The competition combines improvements across three tasks by geometric mean.
If the other two tasks stayed at their reference costs, our current invsqrt
gain alone would yield a hypothetical combined score of **1.063169**. This
is a calculation, not a submission or a measured leaderboard score.

## Decision

There is credible local correctness and a measured improvement over the
initial reference. Current evidence does **not** support a claim of being
close to a prize. We should test RO fairly before attributing search gains
to it. Precision tuning within this fixed family is a bounded experiment;
closing an order-of-magnitude public cost gap likely requires different
arithmetic/phase architecture, which precision tuning alone does not test.

The follow-up protocol compares RO-V2 and RO-auto with random search and
differential evolution, at exactly 1,024 calls for each of ten seeds.
[Protocol and code](../../experiments/vqcc_search/README.md).
