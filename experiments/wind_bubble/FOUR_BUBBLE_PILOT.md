# Four-bubble pilot before a many-source scaling study

## Purpose and fixed setup

Extend the existing online diagnostics and geometric histories to one fixed
four-source interaction. This is a functionality/interpretability pilot,
not a developed turbulent box, a physical-ancestry validation or a benchmark.

- Uniform unit-domain Cartesian grid, 64 cells per axis.
- Four equal winds at `(0.375,0.375,0.375)`, `(0.625,0.625,0.375)`,
  `(0.625,0.375,0.625)`, `(0.375,0.625,0.625)`.
- This regular tetrahedron is noncoplanar. All six source-center distances
  are about 0.353553. The injection spheres, radius 0.0625, are disjoint;
  their minimum boundary clearance is 0.3125 before evolution.
- 17 analysis events, t=0 through t=0.16 at nominal intervals 0.01.
- Existing ambient density 1, pressure 0.01, gamma 5/3 and wind speed 10.
- Each source matches the old equal-two-source reference's prescribed
  discrete mass and mechanical-energy coefficients. At 64 cubed, each mask
  contains 160 cells, requiring a shared mass-parameter scale of 1.1.
  Each prescribed energy coefficient is about 0.32825707; the four-source
  total is twice the original two-source total, not the same total input.

These coefficients are defined before the existing state-dependent EI
pressure conversion. They are not a measurement of net gas-energy deposition.
Source-budget matching does not turn different layouts into a convergence
sequence or fix physical interaction times in advance.

## Implementation

`run_four_bubbles.py` delegates integration, callbacks, diagnostics, sparse
catalogs, slices, histograms and optional tracking/verification to the existing
`run_two_bubbles.run` implementation. The source layout and budget/strength
builders are the new inputs. There is no new integration or analysis path.

`multi_source_setup.py` factors preflight for the new pilot: finite in-domain
source positions, nonoverlapping spheres, actual mask counts and per-source
budget matching. This shared-parameter helper explicitly rejects unequal or
empty mask counts rather than silently changing per-source injection. It is
not yet an arbitrary random-source prescription. Historical two/three-source
helpers and outputs remain unchanged.

The detector/default mode, Mach estimator, surface selection, grouping gates,
maximum-five model and delayed-support rules are unchanged. The previously
validated support-reuse optimization is used by the shared tracker.

Group labels remain geometric continuation segments. Source distances and
source-overlap flags do not assign ownership. Display meshes are separate
postprocessing and are not generated during this pilot run.

## Local checks

The focused suite covers actual four-source EI injection, superposition of
disjoint weighted sources, solver/preflight mask agreement, per-source budget
preservation, total input accounting, source exclusion in sparse catalogs,
shared-runner delegation and existing two/three-source/tracker regressions.

The local 32-cubed, three-event, t-end=0.02 CPU smoke run checks the complete
online/offline and batch workflow. It does not reach the intended collision
regime or replace the 64-cubed GPU pilot.

Completed locally: 51 focused tests pass. The smoke output in
`outputs/four_bubbles_cpu_smoke_n032/` reports all three events passing,
zero finite scalar difference, exact histograms and full tracking records,
matching sparse catalogs/slices, eight group observations, independent batch
agreement and matching final gas states. A fresh-process check also confirms
that the source preflight imports no JAX. The remote pilot is still pending.

NumPy-only source preflight (creates no output directory or JAX computation):

```sh
python -m experiments.wind_bubble.run_four_bubbles \
  --preflight-only --track-shocks --verify-offline
```

## Preserve GPU outputs and synchronize

On the GPU checkout, preserve previous outputs before pulling:

```sh
cd ~/master_thesis
git stash push -u -m "Preserve GPU outputs before four-bubble pilot" -- outputs/
git stash list -1
git pull --ff-only
```

Run the pull only after the new local code has been committed and pushed.
Leave the stash saved after pulling. Reapplying untracked output files over
newly tracked copies can recreate conflicts. This command does not stash
local code edits outside `outputs/`; inspect `git status` if those block a pull.

## compgpu2 launch on a free GPU 0

The following uses the Python 3.12 environment that worked on the older GPU
hosts, rather than assuming `.venv-iwr` resolves to the correct interpreter.
Selecting GPU 0 is not a reservation. Check it immediately before starting.

```sh
export THESIS_PY="$HOME/.venvs/master-thesis-py312/bin/python"
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi -i 0
"$THESIS_PY" -m experiments.wind_bubble.check_gpu_environment

"$THESIS_PY" -m unittest \
  tests.experiments.test_four_bubbles \
  tests.experiments.test_three_bubbles \
  tests.experiments.test_two_bubble_source_budget \
  tests.experiments.test_streaming_history \
  tests.experiments.test_streaming_products

"$THESIS_PY" -m experiments.wind_bubble.run_four_bubbles \
  --num-cells 64 --num-analysis 17 --t-end 0.16 \
  --track-shocks --verify-offline --require-gpu \
  --output-dir outputs/four_bubbles_iwr_compgpu2_n064
```

Launch only after the GPU check and tests pass. Use a fresh output directory;
an existing directory is preserved and the runner refuses to overwrite it.
Console diagnostics appear twice: the online run, then analysis of the
stored-state reference. There is one new online/offline comparison, not a
three-trial performance benchmark.

## Completion and bounded review

```sh
cat outputs/four_bubbles_iwr_compgpu2_n064/equivalence.json

"$THESIS_PY" -m experiments.wind_bubble.plot_two_bubbles \
  outputs/four_bubbles_iwr_compgpu2_n064/online \
  --event-index 16 \
  --output outputs/four_bubbles_iwr_compgpu2_n064/event_0016.png
```

The legacy viewer command supports the multi-source catalog schema and labels
the actual source count. Solid stars on a slice lie in its plane; hollow
stars show off-plane source projections. This is a point-cloud/flow-slice
inspection, not a smooth grouped animation.

First require completed metadata, all 17 events, matching diagnostics,
histograms, catalogs and slices, streaming tracking agreement, independent
batch-reference agreement and matching final gas states. These are execution
checks, not proof of complete physical detection or correct ancestry.

Next inspect a few early/interacting/late events, group histories, uncertain
relationships, sampling flags and detected-surface boundary clearance. A
positive initial injection clearance does not guarantee that evolved fronts
remain far from the boundaries. No source-owner identity is forced to fill
ambiguous groups. Tracker hypothesis limits remain explicit failures.

Do not automatically increase to 128 cubed or eight sources after the run.
Review this single pilot and choose the next step from the observed behavior.
