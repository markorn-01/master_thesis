# Why the recorded collision motion test remained inconclusive

The 128³ GPU simulation completed. All 321 copied strips match their recorded
hashes, and all 137 fingerprinted integration sources match the current
checkout. The failure is in the applicability of the additional strict
state-pair check: it rejects an outer front that is both tracked and detected
by the production finder. Its rejection cannot establish central shock absence.

## The sampled outer states do not satisfy the strict density relation

At each listed time, the upstream sample is six cells ahead of the pressure
crest. The downstream value is the largest recorded density within eight cells
behind it, excluding segments crossing an injection sphere. The prediction
uses pressure at that same downstream sample; it is not a claim that this
sample is a resolved physical downstream plateau.

| Time | Largest recorded density ratio | Ratio predicted from its pressure jump | Shortfall |
| --- | --- | --- | --- |
| 0.04 | 2.085 | 3.881 | 46.3% |
| 0.08 | 2.475 | 3.815 | 35.1% |
| 0.12 | 2.582 | 3.781 | 31.7% |
| 0.15 | 2.702 | 3.741 | 27.8% |

Thus even the local density maximum supplies no clean two-state jump meeting
the strict 10% comparison. Finite-grid sampling of a thin, evolving shocked
shell is consistent with this mismatch. The audit does not isolate every
contribution from numerical broadening, geometry, and unsteadiness. Loosening
the density threshold until this test passes would not validate a new finder.

Nine-event position fits also move less than one cell for much of the run.
At time 0.12, the outer-front speed magnitude is 1.234, 1.084, or 1.020 for
9-, 17-, or 33-event fits. The short fit is sensitive to the crest's position
within a cell. This is another reason to avoid treating the original strict
verdict as a definitive physical test.

## What the measured motion can still tell us

For an ideal-gas compressive shock, upstream flow is supersonic in the shock
frame. With the pressure normal pointing toward higher pressure, the necessary
condition is `(u_n - S_n)/c > 1`. This follows from the Euler shock and
characteristic conditions described in the
[Clawpack Euler chapter](https://www.clawpack.org/riemann_book/html/Euler.html).
Meeting this condition alone does not certify a shock.

The audit samples upstream positions from 0.25 to eight cells away, without
requiring a density ratio or downstream plateau. Source intersections and
crossings of the opposing-flow midplane remain excluded. Position fits use
9, 17, and 33 events. Windows spanning both sides of the collision midplane
are rejected: one fitted speed must not mix incoming and outgoing features
across that plane. There are 38 such invalid fit windows in the recorded paths.

The known outer control and incoming central fronts have clear supersonic
upstream signatures. Later central features in the original x interval
0.45–0.55 are nominally subsonic in all 320 available 17-event checks after
time 0.06; 232 of 233 available 33-event checks are also nominally subsonic.
These counts refer to pressure-feature observations, not physical objects or
complete temporal coverage.

Examples in the left half of the central layer:

| Time | Maximum sampled upstream normal Mach, 17 events | 33 events |
| --- | --- | --- |
| 0.07 | 0.776 | 0.784 |
| 0.08 | 0.603 | Unavailable |
| 0.10 | 0.143 | 0.027 |
| 0.12 | 0.363 | 0.271 |
| 0.15 | 0.803 | 0.798 |

The CSV additionally records a position sensitivity: the worst OLS slope
perturbation if each crest position has an error of half a cell, plus three
fit scatters. This is a declared sensitivity model, not a complete confidence
interval for physical front speed. Some central cases remain subsonic under
this sensitivity, while others cross Mach one and remain inconclusive.

## What can be concluded

The later visible dense band is not established as a resolved shock front by
these measurements. Its measured pressure features generally move too slowly
relative to the sampled upstream sound speed to support that interpretation.
This is additional physical evidence, beyond the production zone rejection.

The actual first collision and any unresolved reflected fronts remain open.
A hidden shock can have different speed and states from the pressure crest
of a mixed grid transition. The data therefore do not justify the statement
“there is no physical shock there.” They also do not justify adding every
central pressure feature to the production shock catalog. A decisive statement
about a hidden front requires better spatially resolved flow evidence.

No production criteria or thresholds were changed, no simulation was rerun,
and the original failed-control verdict is preserved. The audit figure,
per-condition rejection counts, density examples, and sonic sensitivity are
in `outputs/collision_front_motion_iwr_n128/control_failure_audit/`.

## Reproduction

```bash
python -m experiments.wind_bubble.audit_front_motion_control \
  outputs/collision_front_motion_iwr_n128 \
  --output-dir outputs/collision_front_motion_control_audit_new
```

Use a fresh output directory. The audit runs on the CPU using the existing
strips and original assessment. Local validation passed 26 tests covering
the new audit, existing motion and capture logic, and physical jump checks.
