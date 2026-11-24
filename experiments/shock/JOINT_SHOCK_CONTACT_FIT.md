# Joint shock–contact reconstruction: bounded feasibility test

## Question and decision

Can a restricted physical model infer the immediate shocked state when that
state occupies less than a cell and ordinary side sampling measures contact
gas instead?

**Yes for the tested planar model family. This is not yet a replacement finder
or an answer about the central layer in the bubble simulation.** Production
shock-finder code and saved bubble results are unchanged.

The previous density-normal comparison failed because moving the sampling
direction did not recover the correct downstream state. In the principal known
example, the actual shocked density is 3 but the sampled contact density is 24.
This experiment reconstructs the shock and adjacent contact together instead
of interpreting that contact density as the immediate shocked density.

## What the model does

The fitting interface receives only arrays of density, velocity, and pressure,
grid spacing, and common settings. It receives no reference Mach number,
wave positions, contact contrast, or true blur width. Far-state estimates used
to initialize optimization come from the observed ends of the profile and
remain free fit parameters.

The joint model contains three constant regions:

1. Unshocked gas ahead of the shock.
2. Immediate shocked gas between the shock and contact.
3. Contact gas behind that thin slab.

Rankine–Hugoniot relations connect the first two states. Pressure and velocity
are equal across the contact; its density can differ. These are standard
ideal-gas Euler wave relations, as described in the
[Clawpack Riemann book](https://www.clawpack.org/riemann_book/html/Euler.html).
The inverse fitting and acceptance rules here are our experimental implementation,
not a published shock-finding algorithm from that source.

For each trial fan, the model computes exact cell-volume averages of
`U = (rho, rho*u, P/(gamma-1) + rho*u*u/2)`. It optionally applies a conserved-variable
top-hat filter of width 1, 3, or 5 cells. Both orientations and all three widths
are tried without choosing a width from the reference label. Density, momentum,
and energy residuals are normalized by data-derived scales. A data-derived
Galilean frame avoids arbitrary bulk velocities dominating the energy residual.

For the main example, one mixed cell has density 13.5. The fitted interpretation
is `13.5 = 0.5*24 + 0.5*3`: half contact gas and half immediate shocked gas.
No observed cell has the pure shocked density of 3. The fit infers that state
using the pressure/velocity constraints and the assumed fan structure; it does
not create an additional resolved simulation cell.

## Alternatives and safeguards

The same inputs are also fitted with shock-only, contact-only, and constant-entropy
simple-wave alternatives. The latter permits arbitrary observed pressure samples
while constraining density and velocity by entropy and an acoustic Riemann
invariant. It is a specific smooth-compression control, not every possible
smooth flow.

A fit must have a maximum scaled residual at most `1e-3`. Valid models are
ranked with a residual/complexity score and numerical floor `1e-4`; a score
margin below 10 is reported as ambiguous. These choices are common to every
case. The BIC-style score is a heuristic comparison, not a calibrated likelihood
or physical probability.

A usable joint reconstruction also requires Mach at least 1.3, a gap above
0.05 cell, an interior contact, full local Jacobian rank, and agreement of
acceptable restarts across the tested blur/orientation hypotheses. Allowed
spreads are below 5% in Mach and immediate downstream density and below 0.25
cell in positions and gap. Separate fixed-Mach and fixed-gap refits inspect the
main solution. These are finite searches and local checks, not a theorem of
global uniqueness.

## Frozen results

Run:

```bash
MPLCONFIGDIR=/private/tmp/master-thesis-mpl-benchmark \
python -m experiments.shock.check_joint_shock_contact_fit \
  --output-dir outputs/joint_shock_contact_feasibility
```

Use a fresh directory. The recorded run used Python 3.10.20 and NumPy 2.2.5 on
the CPU. The saved inputs, fitting sources, reference sources, and unchanged
production sources have SHA-256 fingerprints in `verification.json`.

| Input family | Cases | Result |
|---|---:|---|
| Half-cell slab, 32 and 64 cells, blur widths 1/3/5 | 6 | Correct joint reconstruction |
| Reflected and bulk-velocity-shifted versions | 2 | Correct joint reconstruction |
| Held-out Mach 2.4, contrast 5, gap 0.35 cell, phase 0.23 | 1 | Correct joint reconstruction |
| Half-cell case with deterministic `1e-4` conserved-variable perturbation | 1 | Reconstruction retained; Mach error about 0.073% |
| Isolated shock and isolated contact | 2 | Simpler model selected; no usable hidden pair |
| Four smooth isentropic compressions | 4 | Isentropic model selected; no usable hidden pair |
| Stationary pressure bump and uniform-pressure compression | 2 | Model mismatch; no usable hidden pair |
| Known shock/contact fan with an unmodeled Gaussian blur | 1 | Model mismatch; no usable hidden pair |

All 19 prespecified checks passed. All 10 positive cases recovered Mach and
immediate downstream density within 5% and both positions within 0.1 cell.
The unperturbed positive cases agreed to floating-point precision. In the
perturbed case, inferred Mach was 2.99782, downstream density 2.99890, and gap
0.500374 cell. No other case received a usable joint-pair claim.

In the main scan, only sampled points at Mach 3 and gap 0.5 cell met the fit
tolerance after refitting the other parameters. This is a discrete conditional
sensitivity check, not a confidence interval. On a shorter 32-cell window,
simple-wave controls can be conservatively labeled ambiguous because the score
margin to a redundant model is smaller; the unit tests require no usable pair.

## What this establishes—and what remains

This gives a concrete alternative to changing the temperature gradient or
sampling farther away: infer a hidden state from a physically constrained
joint model when the necessary far-state information remains available.

However, the positive cases are generated by the same *physical and averaging
family* used in fitting, through an independent reference implementation.
The held-out parameters, reflection, perturbation, and controls check useful
failure modes but do not remove that modeling assumption. The Gaussian example
is a known physical shock that the restricted fit rejects: its best joint fit
has maximum scaled residual about 0.052, exceeding the fixed 0.001 tolerance.
**Model mismatch means this model cannot support a reconstruction; it does not
mean no shock exists.**

Actual curved, unsteady bubble interactions can have varying backgrounds,
multiple waves, and solver-dependent broadening. No bubble profile has been
reclassified here. A subsequent saved-profile feasibility check must preserve
the fixed acceptance rules and permit an inconclusive result. Unrestricted
smooth profiles are not excluded by a good single-snapshot fan fit. Independent
propagation evidence remains necessary for a strong claim about the actual
missing bubble fronts.

The present result justifies considering joint reconstruction. It does not
justify changing production detections or claiming that Tobias's specific
central fronts have been recovered.
