# Saved-bubble jump-screening comparison

## Result

**The validated wider-jump option does not repair the original central gap.**
This comparison separates that result from the successful finite-blast
screening correction. It does not establish absence of physical shocks.

The same captured three-bubble gas was analyzed with the current default
one-cell screen and with `transition_aware_detection=True`, both using the
profile-aware Mach estimator. Five times at each of 64 and 128 cubed were
checked: 0.04, 0.06, 0.08, 0.12 and 0.16. No simulation, source prescription,
grouping, tracking, visualization smoothing, or physical Mach threshold changed.

The inspection window includes both sides of the dense central layer:
x=0.40–0.60, y=0.28–0.47, z=0.42–0.58. The original central audit box is
x=0.45–0.55, y=0.335–0.415, z=0.42–0.58. All reported sample counts here
refer to grid cells in these declared windows, not physical shock objects.

## Central-slice detections

| Time | 64 cubed: baseline → new | 128 cubed: baseline → new |
|---|---:|---:|
| 0.04 | 16 → 16 | 24 → 24 |
| 0.06 | 0 → 0 | 0 → 0 |
| 0.08 | 0 → 0 | 0 → 0 |
| 0.12 | 0 → 0 | 0 → 0 |
| 0.16 | 0 → 0 | 0 → 0 |

Both methods retain the early detected fronts. At 0.08 and 0.16, the entire
central **volume**, not only its plotted slice, has no cells with aligned
density/temperature gradients. The jump-screening extension requires that
alignment and cannot repair its failure. At some inspected pressure features,
the local compression condition fails as well. It would therefore be
incomplete to attribute every late rejection solely to temperature orientation.

The exception is 128 cubed at 0.12: eight central-slice cells pass compression
and alignment. A separate read-only endpoint decomposition finds valid samples
for all eight, but none passes either extended thermodynamic jump threshold.
The largest sampled pressure ratio is 1.2214 versus the required 1.8625; the
largest temperature ratio is 1.2178 versus the required 1.2922. All eight pass
the sampled entropy-increase guard. These are measurements at the algorithm's
chosen endpoints, not independent physical Mach estimates.

The wider window does have peripheral additions/removals. At 128 cubed and
0.16, there are 94 added and 82 removed source-filtered surface cells. None
fills the original central volume, which remains empty. Two added cells have
upstream endpoints inside a prescribed source. These changes must not be
advertised as recovered reflected collision fronts or complete physical
surfaces. At 0.06, some baseline samples are removed by the final Mach cut.

## Frame-aware physical classification

`frame_aware_classification.csv` has one row for each of the ten saved states.
It includes the control status, available measured pressure-feature speeds,
normal vectors, signed shock-frame Mach sensitivity envelopes, sampled entropy
and conservation residuals, characteristic-ordering tests, and snapshot
compression/alignment counts. Detailed feature evidence is saved separately.

Normals for this check come from the three-dimensional **pressure** gradient
estimated from a saved transverse strip. Speeds come from temporally bracketed,
consecutive pressure-feature positions, using 9- and 17-event windows. They are
not inferred by imposing mass balance. The same existing state-sampling,
entropy, sonic-ordering, conservation and endpoint-flatness procedure is applied
to the clean outermost fronts on both sides before interpreting central features.

**All ten classifications are ambiguous under this procedure.** This means
the physical evidence is insufficient, not that every visible structure is
unphysical or that known outer shocks are absent:

- No matching 64-cubed high-cadence history is available, so 128-cubed speeds
  are not transferred to those five states.
- At 128 cubed and 0.04/0.06/0.08/0.12, the outer-front controls fail the strict
  sampled Euler/entropy/endpoint test. A failed positive control forbids using
  the same procedure to declare the central region shock-free. For context,
  the best recorded outer density-ratio residuals are about 29–50%, above the
  fixed 10% gate; other checks remain visible in the table.
- At 0.16 there is no subsequent saved event, so a two-sided temporal speed
  fit is unavailable. No one-sided extrapolation is substituted.
- The 321-event strip run uses a finer diagnostic cadence and is a separate
  numerical evolution. Matching physical parameters do not make its gas arrays
  identical to the original captures. On the matched 128-cubed strips, density
  relative L1 differences are about 0.015–0.118%. Its motion estimates are
  retained as supplemental evidence, not exact velocities for the original gas.

Pressure-ratio Mach values in the table are explicitly **apparent** estimates.
They are not physical shock Mach numbers until frame-aware jump conditions
are supported. Sensitivity envelopes can cross zero because they describe
signed relative normal-flow ratios of trial samples; they are not rigorous
confidence intervals. Missing motion or guarded samples remain blank.

No `resolved shock` or `rarefaction/contact-compatible` label is assigned from a
temperature rejection or noncompressive cell alone. Failed/unavailable controls
force the classification to remain ambiguous. The earlier failed-control
diagnosis is preserved, not replaced with a physical-absence claim.

A subsequent [bounded outer-control calibration](OUTER_CONTROL_CALIBRATION.md)
uses gas and motion from the same high-cadence run at 0.08 and 0.12. Longer
speed fits and quarter-cell endpoint searches do not repair the strict jump
control. All fifteen known-state checks pass, but all eight bubble comparisons
fail. This narrows the limitation without changing these ambiguous labels.

## Integrity and limits

Captured full-state hashes match their original GPU audits. Source-code hashes
are recorded for this replay; historical detector-source changes are recorded
separately rather than pretending corrected sources equal their old versions.
Early compression/alignment/minimum-jump/zone masks match the historical crop
masks. Four historical surface-cell assignments differ from the current
baseline across the two 0.08 windows; no central-slice count changes. Mode
comparisons always use the current baseline and corrected path on identical
gas and the same CPU backend.

The calculation uses a 24-cell halo around the window. Enlarging it to 28 cells
at the early/late times of both resolutions leaves all recorded Boolean masks
unchanged. Float32 interpolation differences are recorded separately; maximum
surface-Mach difference is 1.53e-5. This is a local-window comparison, not full
domain completeness or a new GPU performance benchmark.

The native axes are y=0.3671875, z=0.5078125 at 64 cubed and y=0.37109375,
z=0.50390625 at 128 cubed. Each original/new comparison shares exactly the same
axis. The two resolutions are not claimed to sample one identical physical ray
or establish convergence.

The production default remains unchanged. Further changes to temperature
orientation/alignment require their own known-shock/contact controls and a
physical validation procedure that succeeds on the outer controls. This
comparison alone does not justify relaxing alignment or asserting a missed
above-threshold physical front.

## Reproduce

Use compatible project Python, the captured 64/128-cubed GPU states, and the
existing high-cadence strips. The following commands never integrate the gas.
Use a fresh comparison output directory.

```sh
python -m experiments.wind_bubble.compare_collision_jump_screening
python -m experiments.wind_bubble.inspect_bubble_jump_candidates
python -m experiments.wind_bubble.classify_saved_collision_frames

python -m unittest \
  tests.experiments.test_collision_jump_comparison \
  tests.experiments.test_collision_front_motion \
  tests.physics_modules.shock_finder.test_transition_jump_screening
```

The 27 software tests pass. The physical outer-control failure is a recorded
scientific limitation; it is not converted into a passing physical validation.
