# Evolved planar shock collision: reflected-front control

## Purpose

Check Tobias's physical picture: after two symmetric shocks meet, two reflected
fronts move apart around dense, high-pressure, nearly stationary gas. A central
density maximum is not itself a shock.

This experiment **actually evolves the hydrodynamic equations**. The earlier
`check_colliding_shocks` test supplied exact or synthetically broadened profiles
directly to the finder without time integration. Its reference is reused for
initial conditions and independent scoring, not to replace numerical evolution.
No reference position or expected Mach number is passed to the detector.

The ideal-gas Euler jump/contact relations are described in the
[Clawpack Riemann book](https://www.clawpack.org/riemann_book/html/Euler.html).
The independent `SymmetricShockCollision` reference checks conservation and
entropy conditions at the incident and reflected fronts.

## Fixed setup and scoring

- Six evolved runs: incident Mach 3 and 10, each at 256, 512, and 1024 cells.
- Cartesian 1D hydrodynamics, gamma = 5/3; no wind sources, cooling, extra
  smoothing, or imposed positivity floors.
- Default finite-volume HLL/minmod/RK2 choices and CFL 0.4, as in the hydro
  bubble setup. The snapshots use float32 gas evolution.
- Incident fronts start at x = 0.25 and 0.75. Constant post-shock reservoirs
  supply the approaching gas. Open-boundary inflow is included in the expected
  box-integral balance; this is not a closed-box energy-conservation test.
- 81 saved states through 1.6 times the exact collision time.
- Every evolved line is extruded unchanged onto `(N,8,8)` with equal cell
  widths, solely for the **production 3D finder**. No 3D gas evolution is run.
  Transverse interior masks must agree. A unit test checks thin-slab output
  against the full-cube finder on the same reference profile.
- The original production threshold and profile-aware estimator are retained.

The declared separated regime requires exact outgoing separation >=16 cells
and time/collision-time >=1.1. Each front must lie within two cells of its
independent target, on its own side of the midplane, with the correct normal
sign and Mach error <=5%. One midpoint detection cannot count for both fronts.
All earlier events remain in the tables, including failures. The 16-cell gate
is a conservative control definition, not a universal minimum for detection.

Independent pressure-gradient crests measure outgoing speeds over time ratios
>=1.2, without finder masks. Boundary-flux balance, slab-state medians and
near-zero slab velocity provide a numerical-flow control independently of
detection. Passing detection alone would not validate a wrong hydro solution.

## Finite incident Mach versus the strong-shock limit

| Incident Mach | Reflected Mach | Reflected pressure ratio | Reflected density ratio | Outward speed / incident speed |
|---|---:|---:|---:|---:|
| 3 | 1.914854 | 4.333333 | 2.2 | 0.555556 |
| 10 | 2.201059 | 5.805825 | 2.470297 | 0.505000 |
| Strong-shock limit | sqrt(5), about 2.236 | 6 | 2.5 | 0.5 |

These finite-Mach targets are computed from the exact reference before scoring.
Tobias's quoted values are strong-incident-shock limits, not exact requirements
for an incident Mach-3 collision.

## Completed CPU results

Artifact: `outputs/evolved_planar_collision_n0256_n0512_n1024/`.
Python 3.10.20, JAX 0.6.2, CPU.

| Incident Mach | Cells | Separated events | Both fronts located and Mach within 5% at every required event |
|---|---:|---:|---|
| 3 | 256 | 19 | Yes |
| 3 | 512 | 25 | Yes |
| 3 | 1024 | 26 | Yes |
| 10 | 256 | 18 | Yes |
| 10 | 512 | 24 | Yes |
| 10 | 1024 | 26 | Yes |

All **276 required individual front checks** pass. Maximum Mach error is about
**1.40%**, maximum position error about **0.71 cell**. No midplane surface is
selected in any required separated event.

Independently measured speeds differ from the exact values by at most about
0.36%. Maximum normalized integral-balance error is about 3.4e-6. Final
conserved-variable L1 errors decrease with resolution for both strengths.
At 1024 cells, final median slab density/pressure errors are below 0.12%, and
slab RMS velocity is below 0.003 times the incident post-shock sound speed.
These results support this numerical-flow control, not bubble convergence.

### Preserved early failures

At 256 cells and time ratio 1.02, exact outgoing separations are only about
1.42 cells (Mach-3 case) and 1.29 cells (Mach-10 case). Zone candidates exist,
but only one correctly assigned surface is recovered; its Mach is about 12%
too low. At time ratio 1.06 for Mach 10, both fronts are located but Mach is
about 6.4% too high. These failures remain in `fronts.csv`.

All saved outgoing events at 512 and 1024 cells meet the location/Mach target.
The cadence does not sample arbitrarily close to the meeting instant, so this
does not establish perfect early recovery. Reporting the separated regime
must not hide the lower-resolution failures.

## Incidental native 1D bug

The smoke test found that `_find_shock_surface_1d` used an exclusive loop bound
that omitted the final contiguous zone. One zone produced no selected surface;
two zones lost the last one. The correction changes that bound to
`max_segment_id + 1`. Four regressions cover single/multiple/empty/tied zones;
three failed before the fix and all pass afterward.

This changes **only the native 1D branch**, not the 3D selector used for bubbles.
It does not explain the bubble gap. The main experiment uses the 3D finder on
thin extrusions, so its positive result is not a substitution of the 1D selector.
The full shock-finder suite, including the new tests, passes: **107 tests**.

## Interpretation and next boundary

The existing 3D finder recovers both reflected shocks in a genuinely evolved,
resolved planar collision while leaving the interior slab unmarked. This is
stronger evidence than exact constructed snapshots alone.

It is **not yet a finite-blast or bubble test**. Constant reservoirs do not
model fronts leaving dense shells and entering hot cavities; reflected Mach
does not fade here. Contacts and asymmetry are absent. This does not establish
whether the disputed bubble fronts are resolved, missed or below threshold.

The next distinct extension is a finite blast-wave collision with changing
upstream conditions. Its physics must be assessed independently before treating
missing detections as fading. The original bubble should be examined for two
moving slab boundaries, without requiring a shock at the midplane. A proposed
temperature-orientation explanation remains something to test.

That [finite-blast extension](FINITE_BLAST_COLLISION.md) has now completed.
It finds weakening without a supported below-threshold phase and exposes
above-supported minimum-jump screening misses. This is a separate mechanism
from the original bubble alignment rejection; the distinction is retained.

## Reproduce

```sh
python -m unittest tests.physics_modules.shock_finder.test_shock_surface_1d \
  tests.physics_modules.shock_finder.test_evolved_collision

MPLCONFIGDIR=/private/tmp/master-thesis-mpl-benchmark \
python -m experiments.shock.run_evolved_collision \
  --resolutions 256 512 1024 --incoming-machs 3 10 --events 81 \
  --output-dir outputs/evolved_planar_collision_n0256_n0512_n1024
```

Use a fresh directory and compatible project Python. No GPU is needed. Sources
are fingerprinted before/after execution. Prior archives are not overwritten
or retroactively revalidated against modified source files.
