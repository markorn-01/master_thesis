# Finite blast collision checkpoint

See the [method, controls, results, and limits](../../experiments/shock/FINITE_BLAST_COLLISION.md).

Three genuinely evolved finite-energy runs, 512/1024/2048 cells and 301 events.
Independent pressure-feature motion and sampled-jump evidence are formed without
production masks or Mach estimates, then compared with the production 3D finder
on thin extrusions. Strong/weak/smooth/evolved controls pass.

The measured outgoing candidates weaken, but **no supported crossing below
Mach 1.3 is observed**. Forty repeated above-supported front-events lack a
matched detection. Their local minimum-jump screening fails; compression and
alignment candidates remain present. These are observations, not 40 shocks.

Files:

- `verification.json`: fixed settings, case outcomes, physical scope, fingerprints.
- `controls.json` and `control_*.npz`: independent evidence controls.
- `finite_blast_n*.npz`: evolved rho/u/P, actual times, and production stage fields.
- `observations.csv`: every tracked pressure feature, evidence status, and detector comparison.
- `sampled_jump_checks.csv`: all tested side pairs, including failures.
- `tracks.csv`: conditional track summaries; no invented continuation after loss.
- `resolution_comparison.csv`: conserved-profile refinement comparisons.
- `finite_blast_spacetime.png`: evolving flow, pressure-feature paths, detections.
- `finite_blast_mach.png`: initial quick-look summary; use the reviewed figure
  below for interpretation because it explicitly breaks unverified intervals.
- `review/mach_supported_intervals.png`: supported Mach intervals, with gaps retained.
- `review/missed_front_stages.png`: concrete 2048-cell screening miss near Mach 1.54.
- `review/example.json`, `example.npz`, `above_supported_misses.csv`, `verification.json`:
  exact stage intersections and the selected example, without rerunning the gas.

Jump compatibility is conditional on sampled states and the declared position
sensitivity. A lost feature is not shock absence. This does not settle the
original 3D bubble layer, and no production change is promoted by this result.
