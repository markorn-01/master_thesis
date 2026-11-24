# Known-reference colliding-shock test

This document describes the **frozen pre-correction baseline** in
`outputs/colliding_shocks_reference_n032_n064/`. The opposing-front correction
and its November 24 validation are documented in
[OPPOSING_FRONT_CORRECTION.md](OPPOSING_FRONT_CORRECTION.md). Historical source
hashes and results remain unchanged; running the benchmark on current code
now evaluates the corrected finder.

This benchmark answers a narrower question than the wind-bubble audit:
**can the unchanged production finder recover shocks whose presence,
positions, and strengths are known without using that finder?**

It constructs exact planar Euler states and analyzes them on a 3D Cartesian
grid. It does not evolve the gas, group detections, or reconstruct a display
mesh. Therefore a failure here can be attributed to detection or state sampling,
without depending on the hydrodynamic solver, grouping, or smoothing.

## Independent physical reference

`colliding_shocks_reference.py` uses NumPy and the ideal-gas jump conditions,
without importing the detector. With gamma = 5/3, ambient density and pressure
are 1 and ambient velocity is zero. Two Mach-3 shocks initially at x = 0.25
and 0.75 move toward each other. Their downstream streams have density 3,
pressure 11, and velocities +2.581989 and -2.581989.

The shocks meet at t = 0.06454972. The resulting symmetric Riemann problem
has two outgoing shocks with upstream-frame Mach number 1.914854. The central
gas is at rest, with density 6.6 and pressure 47.666667. The outgoing lab-frame
shock speed is 2.151657. Symmetry gives no nonzero contact discontinuity between
the two central states. At the exact meeting instant there are no two spatially
separated fronts to recover.

For the outgoing branches, if mu is the incoming downstream speed divided by
its sound speed, the jump conditions give

    A = (gamma + 1) mu / 2
    M_out = (A + sqrt(A^2 + 4)) / 2.

The reference is checked by mass, momentum, and energy flux balance at all
four shock branches, entropy increase across both types of shock, and the
integrated box conservation law through the collision. Normalized local flux
residuals are below 3e-16 for the reported Mach-3 reference.

The input stores exact cell averages of conserved density, momentum, and
total energy, then converts them to primitive variables. In a cell containing
a discontinuity, the resulting pressure need not equal the volume average of
the exact primitive pressure. Additional 3- and 5-cell box filters operate on
conserved variables. These are **synthetic broadening controls**, not additional
exact solutions of the Euler equations.

The Euler/Riemann construction follows the standard conservation-law framework
described in the [Clawpack Euler Riemann chapter](https://www.clawpack.org/riemann_book/html/Euler.html).

## Checks and scoring

The recorded CPU/JAX 0.6.2 run contains 96 cases at 32^3 and 64^3:

- Incoming and outgoing pairs at separations 0.5, 1, 2, 4, 8, and 16 cell widths.
- Exact conservative averages and 3-/5-cell broadening controls.
- Isolated incoming/outgoing shocks with the same plateau states.
- A stationary isobaric contact, and the exact meeting instant.

Both fronts must be detected separately, on the correct side of the domain
midpoint, with the correct normal orientation. Coverage is reported per interior
transverse grid ray within 1 and 2 cell widths of each known front; the two-cell
criterion is used in the summary. A single central detection cannot count as
both fronts. Mach accuracy is evaluated separately, using the primary
profile-aware pressure-jump reconstruction and a 5% comparison threshold.
Outer two-cell layers are excluded from scoring. Candidate and surface-cell
counts are grid sample counts, not numbers of physical shocks.

## Findings from the unchanged finder

| Exact-average case | Known physical answer | Observed result |
| --- | --- | --- |
| Isolated shocks and widely separated pairs | Incoming M = 3; outgoing M = 1.914854 | Located with correct Mach numbers |
| Incoming pair, separation 2 cells | Two Mach-3 fronts | Both located, but both reported as M = 1 |
| Outgoing pair, separation 2 cells | Two Mach-1.914854 fronts | Zone candidates present, but no selected surface |
| Stationary contact | No shock | No detected surface for any tested filter |

The separate traces show why the two close-front cases fail:

1. **Sampling crosses the opposite incoming shock.** The physical upstream
   region has P = 1. With two-cell separation, the sampler follows the joined
   Boolean shock zone into P = 11 gas beyond the other front. It obtains
   P_post/P_pre = 11/11 and reports M = 1. At eight-cell separation it samples
   P_pre = 1 and correctly reports M = 3. The production 1D sampler trace is
   checked against the saved production 3D Mach values.
2. **Opposing outgoing fronts reject each other during surface selection.**
   Four consecutive zone cells share the same divergence minimum. All fail
   the non-strict backward comparison. The inner two cells have opposite
   normals and disqualify each other; no representative surface survives.
   The reconstructed comparison operands reproduce the saved 3D surface mask.

Broadening also introduces isolated-shock bias: the five-cell control has
about 5.7% incoming and 12.9% outgoing Mach error even without a second front.
Those errors should not all be attributed to collision overlap.

These tests establish concrete close-front limitations, including a missed
known outgoing pair. They do **not** establish that the wind-bubble density
ridge contains the same structures or fails for the same reasons: the checked
bubble region was rejected earlier, at the shock-zone criteria. This planar,
aligned, symmetric test is not a convergence study of the evolved bubble.
Production physics and thresholds have not been changed.

## Next bounded correction

Investigate normal-direction consistency when comparing candidate surfaces and
following sampling rays through joined zones. Keep the contact-rejection tests.
Opposing-normal ray termination is explicitly discussed in Section 2.3.3 of
[Schaal and Springel](https://arxiv.org/html/1407.4117). That is a reference for a
candidate correction, not evidence that a correction has already been applied
or that it will resolve the separate bubble-zone rejection.

## Reproduce

Use a compatible environment with JAX, NumPy, and Matplotlib, from the repo root.
Each output directory must be new.

```bash
python -m unittest tests.physics_modules.shock_finder.test_colliding_shocks_reference

python -m experiments.shock.check_colliding_shocks \
  --output-dir outputs/colliding_shocks_reference_new

python -m experiments.shock.trace_colliding_shock_sampling \
  --root outputs/colliding_shocks_reference_new \
  --output-dir outputs/colliding_shocks_trace_new
```

The trace expects the default 64^3 cases. The benchmark accepts `--require-gpu`
for optional backend confirmation; GPU execution is not required for this
analytic-state experiment. Source fingerprints are saved and checked. The
trace requires the benchmark sources to remain unchanged since the run.
The historical failure trace additionally expects the pre-correction behavior;
use revision `07a8ec16a32ee50a3f444008942594d68aea54f8` in a separate checkout
to reproduce that baseline. For current code, use the corrected benchmark and
comparison commands in `OPPOSING_FRONT_CORRECTION.md` instead.

Recorded artifacts: `outputs/colliding_shocks_reference_n032_n064/` and
`outputs/colliding_shocks_sampling_trace_n064/`.
