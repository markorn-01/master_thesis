# Known shock beside a contact: unchanged-finder diagnosis

This experiment tests whether a physical shock can be rejected by the
temperature-based shock-zone criteria when a contact discontinuity lies nearby.
It adds a physical reference and diagnostics; **no production finder, threshold,
time integrator, grouping, or display code is changed**.

## Independent physical solution

The exact ideal-gas Euler Riemann fan consists of a contact followed by a
right-going shock; the left acoustic wave has zero strength. A contact can
have different densities but must have the same pressure and velocity on both
sides. The shock satisfies all three Rankine–Hugoniot conservation laws and
increases entropy. This is the standard Euler wave structure described in the
[Clawpack Euler chapter](https://www.clawpack.org/riemann_book/html/Euler.html).

For gamma = 5/3 and shock Mach 3, the states in left-to-right order in the
dense-contact example are:

| Region | Density | Velocity | Pressure | Temperature proxy P/rho |
| --- | --- | --- | --- | --- |
| Gas left of contact | 24 | 2.581989 | 11 | 0.458333 |
| Shocked gas between contact and shock | 3 | 2.581989 | 11 | 3.666667 |
| Undisturbed gas right of shock | 1 | 0 | 1 | 1 |

The contact moves at 2.581989 and the shock at 3.872983 in these units. The
physical shock heats gas from T = 1 to T = 3.666667, regardless of the contact.
Their increasing separation is fixed by the difference of their speeds.
This is a nearby shock/contact test, not a simulation of their collision.

`shock_contact_reference.py` constructs the fan in NumPy without using the
finder. It supplies exact averages of conserved density, momentum, and energy,
then converts them to primitive variables. Pressure or temperature derived
from a mixed cell is not generally its exact primitive volume average.

The reference is checked by:

- Mass, momentum, and energy flux balance at the shock and contact.
- Increasing entropy across the shock; equal pressure and velocity across the contact.
- Agreement with the existing general exact Riemann pressure solver, in float64,
  for both lighter and denser contacts.
- Global conserved-box balance while both waves remain within the box, and
  consistency of the prescribed positions, time, grid phase, and reflection.

## Bounded test matrix

`check_shock_contact.py` analyzes 3D planar inputs with the current production
finder and exposes its actual stage masks. The run contains 228 cases:
180 pairs, 12 isolated shocks, and 36 isolated moving contacts.

- Grids: 32^3 and 64^3; shock Mach: 3.
- Contact density factors relative to the immediate downstream gas: 0.25, 4, 8.
- Shock–contact separations: 0.5, 1, 2, 4, 8 cell widths.
- Shock on a cell edge or shifted by half a cell.
- Exact conservative averages (filter width 1) and synthetic 3-/5-cell
  averaging of conserved fields.

The filtered inputs are controlled broadening experiments, not further exact
Euler solutions or measured hydrodynamic solver profiles. The grid-scaled
matrix by itself is not a fixed-physical-geometry resolution study.

The independent pressure-transition footprint contains centerline cells with
a nonzero centered pressure difference. This reference has only one physical
pressure jump: the contact has constant pressure. A tolerance removes only
reference arithmetic roundoff. Pass/fail counts within that footprint use the
actual production masks, including joint Boolean patterns.

Location scoring requires an interior surface with the correct normal sign
within two cells of the known shock. For pairs it must additionally lie on
the shock's side of the shock/contact midpoint; a nearby contact detection
cannot stand in for the shock. Mach accuracy is scored separately at 5%.
The table also retains counts before this identity gate so displacement is
not confused with the absence of any candidate. Counts are samples, not
physical shock-object counts. This matrix uses x-aligned right-going waves;
unit controls additionally check reflection.

## What the results establish

All isolated exact shock controls are recovered, and no isolated moving contact
produces a surface. For a density-factor-8 contact, an edge-aligned shock only
half a cell away has **no shock-zone candidates**, at both tested grids.
The corresponding synthetic 3-/5-cell filters also produce no candidates.
The complete matrix has six pair cases rejected at this earlier stage.

In the exact unfiltered example the two pressure-transition cells are both
compressive. Both fail temperature–density alignment and the pressure-jump
test. The adjacent hot slab is narrower than a cell and is absent as a hot
plateau in the grid input. The contact-containing cell has density 13.5,
pressure 11, and T = 0.814815. The next upstream cell has density 1, pressure
1, and T = 1. Thus the grid temperature increases to the right while density
decreases. The computed normal points left, opposite to the physical shock's
hot-to-cold direction. The measured detection pressure jump becomes
Delta ln P = -ln(11) = -2.397895 instead of +ln(11).

This is a **known physical shock whose finite-grid representation is rejected
by the existing zone criteria**. It is not evidence of a fully resolved shock
being missed, and does not justify removing contact rejection. Grid phase
matters: a half-cell shift can produce a detection with biased Mach instead
of complete rejection. Other close pairs also show state-sampling bias even
when a surface survives. The full 180-pair matrix has 74 recovered, 96
Mach-biased, 6 zone-rejected, and 4 direction/location-limited cases. These
counts characterize a constructed matrix, not an astrophysical accuracy rate.

## Same physical fan, different resolution

`check_shock_contact_resolution.py` holds the physical separation at 1/64,
the shock at x = 0.5, and the contact at x = 0.484375. The physical time and
Riemann origin also remain fixed. No synthetic filtering is used. Conservative
averages at the coarser grids are checked against grouped averages of the
128-cell reference, so this is genuinely the same physical solution.

| Grid | Separation in cells | Detector result |
| --- | --- | --- |
| 32^3 | 0.5 | No zone candidates; no detected shock |
| 64^3 | 1 | Detected, Mach 2.236 instead of 3 |
| 128^3 | 2 | Detected, Mach 3 |

Increasing resolution therefore restores correct recovery in this particular
reference. This does not guarantee that a particular increase in resolution
will restore the fronts in the evolved bubble simulation.

## Meaning for the bubble question

The test supplies a concrete mechanism for the earlier rejection seen in the
bubble audit: an unresolved shock/contact combination can reverse the measured
thermal direction and fail the zone criteria even when a physical shock is
present. It strengthens the warning that absence of detections does not prove
absence of a physical shock.

It does **not** identify the bubble's central layer as this exact configuration.
Its curved, interacting, nonuniform flow still needs independent physical
evidence. The previously saved bubble recheck remains valid: its later central
region has no candidates at 64^3/128^3 with the unchanged criteria. This new
experiment is neither a new bubble run nor a new correction.

## Validation and reproduction

The complete shock-finder suite passed locally: 71 tests, including 11 new
reference/control tests. JAX 0.6.2 on CPU was used. GPU confirmation of these
new experiments is outstanding. Sources are fingerprinted and the historical
opposing-front results remain unchanged.

```bash
python -m unittest discover -s tests/physics_modules/shock_finder -p 'test_*.py'

python -m experiments.shock.check_shock_contact \
  --output-dir outputs/shock_contact_reference_new

python -m experiments.shock.check_shock_contact_resolution \
  --output-dir outputs/shock_contact_fixed_gap_new
```

Both experiment commands accept `--require-gpu`. Use a fresh output directory.
Recorded results are in `outputs/shock_contact_reference_n032_n064/` and
`outputs/shock_contact_fixed_gap_n032_n064_n128/`. The most direct explanation
is the fixed-gap figure, which overlays the exact hot slab with each grid's
input and detected zone.
