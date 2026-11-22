# Small callback-integrated tracking check

Completed locally with Python 3.10.20, JAX 0.6.2 and the **CPU** backend.
The source hashes in `metadata.json` identify the tested working-tree code;
the recorded commit is its parent revision before committing this change.
This is not remote GPU validation or a performance benchmark.

```sh
JAX_PLATFORMS=cpu python -m experiments.wind_bubble.run_two_bubbles \
  --num-cells 32 --num-analysis 17 --t-end 0.16 \
  --track-shocks --verify-offline \
  --output-dir outputs/online_streaming_cpu_n032
```

Use a fresh directory when reproducing. Two equal sources use the runner's
existing prescription, without cross-resolution budget matching. The 32-cubed
grid is an integration check, not a physically converged collision experiment.

## Results

- All 17 matching-time diagnostic comparisons passed: 21 scalar fields per
  event, maximum finite scalar difference zero, exact histogram counts, matching
  sparse catalogs and background slices, and final gas-state agreement within
  relative/absolute tolerances of 1e-6/1e-8.
- Online and offline incremental products matched; complete JSON records were
  exact. There were 25 retained group observations and five continuation segments.
- Independent whole-sequence batch execution matched all 17 partitions and node
  statistics, 32 adjacent results and 64 delayed/terminal results. The latter
  count includes four terminal pending records, not 64 physical relationships.
- The event graph contained 20 shared continuations, one shared combination and
  two distance-specific continuations. These are geometric algorithm decisions,
  not a ground-truth assessment of physical shock ancestry.
- Between events the tracker retained at most two geometry catalogs. During
  processing it used at most three; their maximum reported array footprint was
  88,928 bytes. Successful close released the retained geometry and interval
  buffers. This excludes temporary arrays, Python objects, caller inputs and
  runtime/solver memory; it is not a process-memory measurement.
- Group CSV rows reproduced their event records, and row/position fingerprints
  matched the respective saved catalogs. All recorded source hashes matched the
  tested code after the run.

The 56-test focused local suite passed, including nine new product/integration
tests and 47 existing component/source regressions. See
`experiments/wind_bubble/ONLINE_GROUP_TRACKING.md` for product definitions,
failure handling, scope and the pending `.venv-iwr` GPU command.
