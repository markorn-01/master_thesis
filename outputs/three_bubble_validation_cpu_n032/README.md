Three-source functional validation, Python 3.11 / JAX 0.10.2 CPU.
Three events through 0.02 at 32³; this is not the collision experiment.
Online/offline scalar differences are zero, and histograms, three-source sparse
catalogs, background slices and final gas state agree (`equivalence.json`).
The shared two-source runner retains default two-source behavior; all 120 saved
NPZ fields in the separate two-source regression match the old baseline exactly.
32 source/injection/budget/audit tests pass.

Sources: (.375,.375,.5), (.625,.375,.5), (.5,.625,.375). Non-collinear,
disjoint, binary-exact coordinates give equal-sized masks. Three equal-strength
winds each match the prescribed per-source rate of the original equal two-wind
reference. Total prescribed mass/energy coefficients are 1.5 times that old
case; this is a new geometry, not an isolated source-count/convergence study.
The EI pressure conversion remains state-dependent. No solver, source physics,
shock estimator, detection threshold or tracking rule was changed.

The viewer supports all three source markers; hollow 2D stars identify off-plane
projections. The third source does not lie on the saved central z slice.
The early low-resolution smoke has all 452 final downstream endpoints inside
source regions, so its Mach/heating measurements are not accepted as physical
validation. Flags are retained, not filtered. The smoke tracking invocation
checks format compatibility only: the two fixed gates do not agree on all early
continuations at this coarse cadence/resolution.

`preflight_n064.json` records the prepared GPU budget: 160 injection cells per
source and prescribed energy coefficients [0.32825707,0.32825707,0.32825707].
The longer 64³ GPU collision run and its .venv-iwr tests remain pending.

After committing/pushing, pulling on the GPU host and selecting a free GPU:

```sh
python -m unittest tests.experiments.test_three_bubbles tests.experiments.test_two_bubbles tests.experiments.test_unequal_source_injection
python -m experiments.wind_bubble.run_three_bubbles \
  --num-cells 64 --num-analysis 17 --t-end 0.16 \
  --verify-offline --require-gpu --output-dir outputs/three_bubbles_iwr_n064
cat outputs/three_bubbles_iwr_n064/equivalence.json
```

The source budget reference and three unit strengths default to the recorded
setup. Use a fresh output directory. No GPU performance or collision/tracking
accuracy claim follows from this smoke test.
