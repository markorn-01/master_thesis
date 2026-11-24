# Direct motion check of the lower collision fronts

The previous tests inferred a shock speed from each pair of gas states.
That speed cannot independently establish that the proposed front moves as
a shock. The saved 17-event background slices also lack velocity components.
This experiment supplies the missing measurement: frequent primitive-flow
profiles from the actual three-bubble simulation, with the production masks
recorded at the same times.

## One focused run on the available GPU

Use 128³ cells, the original three sources and matched prescribed source
budgets, and the original solver timestep cap. Save 321 events from t=0 to
0.16, separated by 0.0005. Dense output adds integration event boundaries, so
this is not a same-state replay of the previous sparse schedule.

Only a strip of shape `5 × 128 × 4 × 4` is transferred per event: density,
three velocities, and pressure. The strip surrounds y=0.375, z=0.5 through
the lower two sources. Its primitive payload is about 13.1 MB before
compression for the entire run. Production zone/surface masks are calculated
on the full grid and only their corresponding small strips are saved. No
complete gas snapshots, full Mach arrays, grouping, or meshes are stored.
The runner is designed for the available 12 GiB GPU; its actual GPU memory
use remains to be checked on that device.

After choosing a free GPU and setting `CUDA_VISIBLE_DEVICES`, run from the
repository root with the working Python 3.12 environment:

```bash
"$HOME/.venvs/master-thesis-py312/bin/python" \
  -m experiments.wind_bubble.run_collision_front_motion \
  --num-cells 128 --num-events 321 --t-end 0.16 \
  --require-gpu \
  --output-dir outputs/collision_front_motion_iwr_n128
```

Use a fresh output directory. Postprocessing runs automatically. If only
postprocessing needs a repair, reuse the completed integration:

```bash
"$HOME/.venvs/master-thesis-py312/bin/python" \
  -m experiments.wind_bubble.run_collision_front_motion \
  --analyze-only \
  --output-dir outputs/collision_front_motion_iwr_n128 \
  --analysis-dir outputs/collision_front_motion_iwr_n128/review_retry
```

## Matched 256-cubed resolution check

The runner also accepts `--num-cells 256`. Keep `--num-events 321 --t-end 0.16`
to match the completed high-cadence 128-cubed schedule, fixed physical injection
radius, and prescribed source budgets. Source coefficients are adjusted using
the actual discrete injection masks; the physical budget is not multiplied
by eight. The primitive strip payload is about 26.3 MB before compression.
The first full-mask 256-cubed attempt on compgpu9 failed with GPU out-of-memory
while requesting an additional 5.18 GiB. Small saved strips do not imply small
temporary allocations: the default diagnostic calculates masks on the full grid.
The revised retry uses `--flow-only` to skip all full-grid detection stages.
It saves the same primitive profiles, times, and physical source budgets.
The GPU cost of hydrodynamic evolution remains; fitting the revised run into
40 GiB has not yet been demonstrated.

After checking that GPU 1 on compgpu9 is free and the environment works:

```sh
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
"$HOME/.venvs/master-thesis-py312/bin/python" \
  -m experiments.wind_bubble.run_collision_front_motion \
  --num-cells 256 --num-events 321 --t-end 0.16 \
  --flow-only --require-gpu \
  --output-dir outputs/collision_front_motion_iwr_compgpu9_n256_flow_only
```

The immediate comparison is the outer-front profiles at 0.08 and 0.12,
followed by the two edges of the collision layer. An inconclusive automatic
assessment is a physical-validation result, not an integration failure.
This run is not online/offline equivalence or a promise that the strict
sampled-state check will pass at higher resolution.

Flow-only metadata records `production_masks_retained=false`, and the NPZ files
omit `production_stages`. The reader reports production comparison as unavailable
and does not turn absent masks into false detections or claimed finder misses.
With this mode, only physical flow/profile assessment is available; production
mask assessment needs a separate capture if later justified.

A local paired CPU smoke check at 32 cubed, nine events to t=0.02, confirmed
exact equality of complete final gas arrays, event times, and every saved flow
strip between default and flow-only modes. The test also checks that flow-only
never invokes `stage_fields`, and exercises the actual flow-only reader.
Eighteen targeted runner/front-motion tests pass. These checks verify the
implementation; they are not a 256-cubed GPU memory measurement.

## What is measured

Pressure-gradient peaks propose short front paths. Adjacent-event links
require matching gradient direction and mutual-nearest spatial support;
ambiguous associations terminate a path. Nine consecutive observations give
a local position-versus-time fit. This speed is measured before the jump
states are selected.

The small transverse stencil supplies the pressure normal. The measured
x crossing speed is projected onto that normal. The profile must be close
to x-normal (`|n_x| >= 0.8`); strongly tilted fronts need a different strip.
Upstream and downstream endpoints are selected independently from the
declared 1, 2, 3, 4, 6, and 8-cell distances. Windows may not cross a source
sphere, the opposing-flow midplane, or a second reversed pressure transition.

Mass flux, normal momentum flux, and total enthalpy are tested in the frame
moving at the measured speed. Density ratio, tangential velocity, entropy
increase, sonic ordering, and endpoint stability are also checked. The
relations follow the ideal-gas Euler shock conditions described in the
[Clawpack Euler chapter](https://www.clawpack.org/riemann_book/html/Euler.html).
The 5%, 10%, and 20% residual settings are experimental accuracy gates, not
universal constants. The fitted-speed scatter is not a complete uncertainty
estimate. Five consecutive accepted events are required for persistent
evidence. Late central evidence is reported separately for t >= 0.06.

## Required outcome checks

- **Outer shock control fails:** central non-detection is inconclusive.
- **Persistent central front passes motion and jump checks, without a nearby
  production surface, above Mach 1.3:** evidence of a missed physical front
  on this ray. All accepted sampling choices must place its Mach above the
  configured threshold before this stronger interpretation is used.
- **Persistent physical front below Mach 1.3:** report it separately as a weak
  front below the production detection setting, not a demonstrated failure
  to detect an above-threshold shock. Sampling choices that straddle 1.3
  remain explicitly uncertain.
- **Persistent central front passes and has nearby production surfaces:**
  the new recorded run supplies a detected front; inspect its time and the
  original requested region before attributing the earlier visual gap.
- **Outer control passes, but no central front passes:** no certified central
  front in the sampled strip. This does not establish shock absence in the
  continuum solution or elsewhere on the interface.

“Nearby” production support means a surface cell center within two grid-cell
widths in 3D; it is not equivalence of refined surface positions. No physical
shock label is assigned from a pressure ridge alone, and no production output
is replaced by this experiment.

## Local checks before the GPU run

The exact space-time controls at 128 cells recover a moving Mach-3 shock,
a moving Mach-1.2 shock, and both outgoing shocks of the exact symmetric
collision. They reject an advected isobaric contact and a translated smooth
isentropic compression. These are known reference sequences, not numerical
hydrodynamic convergence tests. The control figure is in
`outputs/collision_front_motion_controls_n128/motion_controls.png`.

Both a short 32³ smoke integration and a full 321-event 32³ CPU pilot completed.
The pilot does not pass the outer-control gate; it is explicitly inconclusive
at that coarse resolution. These runs verify strip capture, full-grid mask
sampling, output ordering, and automatic review. They do not validate the
128³ bubble interpretation. The actual 128³ GPU run is outstanding.

All 31 targeted motion, capture, physical-jump, resolution-planning, and
mask-audit tests passed locally. The compact validation record is
`outputs/collision_front_motion_validation/validation.json`. Raw CPU test
strips were verified and moved to the temporary archive paths recorded there,
keeping hundreds of temporary test files out of the Git working tree.
