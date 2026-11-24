# Evolved shock/contact comparison: HLL versus HLLC

## Result and decision

**Changing HLL to HLLC gives only a modest improvement in this thin-shell
control. It does not meet the declared criterion for another bubble trial.**
No hydrodynamic solver implementation, production default, detection criterion,
Mach estimator, or threshold was changed. This is an experiment with two
already available solver options, not a proposed replacement of Astronomix.

Eight one-dimensional cases were evolved locally on CPU: shock-plus-contact
and pure-contact problems, HLL and HLLC, at 128 and 256 cells. Each has 201
saved events over 0 <= t <= 0.20. The unchanged 3D finder was applied to thin
extruded slabs at 41 declared events per case. This gives 328 snapshot checks,
including 164 pure-contact negative controls.

## Independent physical reference

The established `shock_contact_reference.py` supplies an exact Euler fan with
a contact and a right-going shock; the left acoustic wave has zero strength.
`shock_contact_solver_reference.py` only rescales that reference to wind-like
units and an initial interface position of 0.625. The original reference's
interfaces and tests are preserved.

The upstream gas has density 1, pressure 0.01, and velocity 0, with gamma=5/3.
The exact shock Mach number is 8. Its downstream state is approximately
(density, velocity, pressure) = (3.82090, 0.76249, 0.79750). The gas on the
other side of the contact has density 0.04 but the same velocity and pressure.
The shock speed is 1.03280; the contact speed is 0.76249. Their separation
therefore grows at 0.27030 in domain units per unit time.

The reference satisfies all three conservative jump conditions at both
interfaces to about 1.5e-16. Its point values also agree with the separate
exact Riemann solver in `astronomix/test_setups/reference_solutions` to
double-precision rounding. Comparisons use exact averages in **conserved**
variables, converted to primitive variables, not pointwise samples as a
substitute for finite-volume averages.

The Euler wave structure follows the
[Clawpack Euler chapter](https://www.clawpack.org/riemann_book/html/Euler.html).
HLLC's restoration of the contact wave is described by
[Toro, Spruce, and Speares (1994)](https://doi.org/10.1007/BF01414629).
Those results motivate the comparison; they do not guarantee an improvement
for this implementation and problem.

## Numerical comparison

Only the configured Riemann solver differs. Initial states, resolution,
minmod limiter, reconstruction stencil, RK2 time integrator, CFL=0.4,
open boundaries, and requested output times are identical.

At t=0.08 the exact plateau density is 3.82090:

| Cells | Layer thickness | HLL peak density | HLLC peak density | HLL peak error | HLLC peak error |
|---|---:|---:|---:|---:|---:|
| 128 | 2.77 cells | 2.6967 | 2.7701 | 29.42% | 27.50% |
| 256 | 5.54 cells | 3.3739 | 3.4058 | 11.70% | 10.86% |

The median peak-density error over the declared t=0.04, 0.08, and 0.16
comparisons decreases by about 6.5% at 128 cells and 7.1% at 256 cells.
Neither reaches the practical 20% error-reduction requirement for a later
bubble trial. This 20% criterion is a trial-selection rule, not a detector
threshold or a universal numerical-accuracy requirement.

All 164 pure-contact checks produce zero detected surfaces. The late
33-event independent jump check passes at some times for both solvers:
13 versus 20 of 80 late events at 128 cells, and 64 versus 64 at 256 cells,
for HLL versus HLLC. A passing fit depends on time, separation, and grid phase;
it is not guaranteed for every thin-layer snapshot.

## What this establishes about the failed outer control

The physical shock exists by construction. Nevertheless, at 128 cells and
t=0.08 the strict sampled-state jump check fails for both solvers while the
production finder locates the front and estimates Mach numbers of about
7.978, close to the exact value 8. The density peak is still about 28–29% low.

Across the twelve selected combinations of solver, grid, and t=0.04/0.08/0.16,
all fronts are located and Mach errors are below 1.1%. Across all positive-time
snapshot checks, 160 of 160 fronts are located, but six Mach errors exceed 5%.
Those six cases have shock/contact separations below 0.35 cell; the largest
Mach error is about 19.4%. The experiment does not claim uniformly accurate
reconstruction of subcell-separated interfaces.

**A failed Rankine-Hugoniot fit to finite-grid samples cannot establish that
a shock is absent from a thin mixed layer.** The equations remain valid at
the physical discontinuity; the sampled cells need not recover its limiting
states. This control directly demonstrates that distinction. The similarity
to the bubble density shortfall supports a numerical-broadening explanation
as a possibility, not a diagnosis of the bubble's physical wave structure.
It does not prove that the bubble finder is complete or that its central
non-detection is correct.

## Conservation and boundaries

Every evolution stays finite with positive density and pressure. The largest
sampled conservation-budget residual is about 1.1e-5, using the **actual
sampled open-boundary Euler fluxes** and trapezoidal integration in time.
This is a resolved-time boundary-budget check, not an audit of every internal
numerical flux or an exact discrete conservation identity.

The maximum discrepancy from the exact solution's total conserved content
is separately retained and reaches about 0.164%. Numerical disturbances
reach the open boundary late in the run, so substituting the exact constant
boundary flux into the numerical conservation budget would conflate
conservation with solution accuracy. Neither discrepancy is hidden by
changing the shock acceptance criteria.

## Products and reproduction

`outputs/shock_contact_hll_hllc_n128_n256` contains:

- Eight compressed archives with evolved and exact cell-average profiles.
- `snapshots.csv`, `motion_checks.csv`, `stability.csv`, and `solver_configs.csv`.
- `shock_contact_density.png`: the thin dense layer at the three selected times.
- `pure_contact_density.png`: the contact-only negative controls.
- `density_and_mach_errors.png`: density-peak error versus Mach error;
  the dashed guides are not the full independent acceptance test.
- `verification.json`: reference values, decisions, scope, source hashes,
  and hashes of the saved numerical cases.

Twenty-six targeted software tests pass. These include exact flux and
finite-volume checks, the independent Riemann oracle, actual evolution with
both solver options, pure-contact rejection, measured-speed controls, and
the unchanged default pressure-ridge windows. Only this experiment expands
the independent ridge search to the full interior domain, because its
single known front is not restricted to the bubble-specific windows.

```sh
python -m experiments.shock.compare_shock_contact_solvers
python -m unittest tests.physics_modules.shock_finder.test_shock_contact_solver_comparison tests.experiments.test_collision_front_motion tests.physics_modules.shock_finder.test_shock_contact_reference.ShockContactPhysicsTests
```

Use a fresh output directory. This completes the bounded solver comparison.
It does not justify another HLLC bubble run as the next experiment on this evidence.
