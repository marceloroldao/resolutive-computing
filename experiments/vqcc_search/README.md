# Equal-budget VQCC precision search

This prospectively specified experiment compares existing, unmodified engines
from Resolutive Computing with conventional search on the same circuit family.
It does not redesign the family or claim a competitive leaderboard result.

## Fixed protocol

- Methods: RO-V2, the public API's RO-auto regime router, uniform random
  search, and SciPy differential evolution.
- Seeds: integers 0 through 9.
- Budget: exactly 1,024 objective calls per method per seed, including one
  shared incumbent evaluation and all repeated decoded configurations.
- Search variables: delta, coefficient, square, product and result fractional
  widths. Inclusive bounds: [11,18], [14,19], [25,31], [25,31], [26,32].
- All methods operate on [0,1]^5, mapped by nearest-integer rounding. This is
  a continuous-to-discrete adaptation, not a native discrete optimizer test.
- Every method receives the native-verified incumbent (14,17,28,28,29).
- RO engines keep their default parameters. The router requires at least
  1,000 calls; 1,024 makes that requirement compatible with the shared start.
- Any unused call budget is filled by a predeclared uniform-random padding
  policy. Excess evaluations are prevented by a common objective wrapper.
- Differential evolution: popsize=7, immediate updating, no polishing,
  zero convergence tolerances, and the same strict call cap.
- Training fitness: model-based Q*C including rotation synthesis, using fixed
  random inputs, interval boundaries and previously known worst inputs.
- The training sample is identical for all methods and independent of search
  seeds. It is not labelled a complete correctness check.
- Validation: take one training-selected configuration per method/seed and
  evaluate the integer model on all 3,133,452 inputs. Validation outcomes do
  not feed back into search. If selection fails or costs more, keep the
  known incumbent, consistently for all methods.
- Report mean, median, standard deviation, best, worst, invalid selections
  and paired RO wins/ties/losses for each conventional baseline.
- The global best full-model configuration receives a fresh 64-shot-per-input
  native verification only if it improves on the already verified incumbent.
  Otherwise no redundant expensive native run is performed.
- The complete integer model must first reproduce the old measured native
  cost to within 1e-6.

All objective calls count toward the fixed budget, including cache hits.
Caches are shared to reduce computation. Search wall times therefore cannot
support comparative speed claims. Per-seed results cannot establish general
optimizer superiority; this is one small, quantized circuit family.

The native verifier comes from the same two pinned public mirrors as the first
experiment, with unchanged hashes and its original license. An authenticated
Kaggle comparison remains pending. No Kaggle submission or rule acceptance is
performed by this experiment.

## Running

The VQCC equal-budget search workflow runs on pushes that change this folder.
It installs pinned NumPy/SciPy and builds native Google kickmix at the same
commit as the first run. Its wheel is cached for subsequent runs.

    python -m unittest discover -s experiments/vqcc_search -p test_protocol.py -v
    python -u experiments/vqcc_search/search.py

The artifact contains the per-seed records, full-domain integer results, source
provenance and any newly native-verified circuit. Passing budget-accounting
tests is not evidence of quantum circuit correctness.
