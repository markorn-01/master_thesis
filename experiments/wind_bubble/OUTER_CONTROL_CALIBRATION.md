# Outer-front control: bounded calibration

## Result

**The strict sampled-state control was not repaired.** Longer pressure-crest
speed fits and finer endpoint searches do not make it pass at either t=0.08
or t=0.12. This closes the bounded speed-versus-sampling check; it does not
settle whether the original central collision fronts are missed physical shocks.

The production finder already detects the tested outer fronts. The failing
test here is an additional independent physical check, not production detection.

## What was checked

All gas and motion use the same completed 128-cubed, 321-event strip run
`outputs/collision_front_motion_iwr_n128`. No speeds are transferred to the
different 17-event simulation. The physical ray is y=0.375, z=0.5, interpolated
within the saved transverse strip. The outermost left pressure-gradient crest
is evaluated at two times with 9-, 17-, and 33-event two-sided motion fits.
These span 0.004, 0.008, and 0.016 simulation-time units.

The existing jump tests are unchanged: independent measured speed, compression,
entropy increase, shock-frame characteristic ordering, mass/momentum/enthalpy
compatibility, density-jump compatibility, tangential velocity, and endpoint
flatness. The original sampling distances are 1, 2, 3, 4, 6, and 8 cell widths.
One additional search at each time uses quarter-cell increments from 1 to 8
cell widths, retaining the same source/domain/opposing-gradient guards. More
interpolation points provide no additional spatial information.

## Measurements

| Time | Speed, 9 events | Speed, 17 events | Speed, 33 events |
|---|---:|---:|---:|
| 0.08 | -1.25986 | -1.23471 | -1.23688 |
| 0.12 | -1.23394 | -1.08415 | -1.01994 |

The first time changes by about 1.9% between the shortest and longest fits;
the second changes by about 21.0%, relative to the longest fit. Fit scatter
is not a complete uncertainty bound. A longer fit averages motion over a
longer interval and is not automatically the instantaneous physical speed.

All eight time/window/sampling comparisons fail the original 10% gate and
also the existing 20% sensitivity gate. The quarter-cell search retains 174
guarded pairs at t=0.08 and 145 at t=0.12; neither yields a compatible pair.
The best pairs still have largest recorded errors of about 33.9% and 29.4%,
respectively, excluding the separately applied entropy-signal gate.

There is also a mismatch independent of fitted speed. At the downstream
density maximum within eight cells, with upstream sampling six cells ahead:

| Time | Recorded density ratio | RH ratio from sampled pressure ratio | Shortfall |
|---|---:|---:|---:|
| 0.08 | 2.4747 | 3.8145 | 35.1% |
| 0.12 | 2.5818 | 3.7812 | 31.7% |

The density peak is narrow and followed by a steep decline rather than a
well-resolved flat downstream density state. These examples illustrate the
state mismatch; they are not an exhaustive proof about every possible sample
or a diagnosis of its unique physical/numerical cause. No theoretical density
is substituted for the recorded gas to force a pass.

## Controls and limits

Fifteen known-state checks pass: incident Mach-3 shocks, reflected shocks,
weak Mach-1.2 shocks, contacts, and smooth isentropic profiles, each with
three speed windows. The weak shock is physically admissible under this check
but remains below the production Mach-1.3 threshold. The isentropic profile is
a constructed negative control, not an independently evolved Euler solution.
Thirteen software tests pass, including the existing front-motion tests.

This demonstrates that passing ideal controls does not calibrate the strict
pointwise jump test for this recorded thin curved shell. Numerical broadening,
transient evolution, geometry, and contact proximity may all matter; their
individual contributions are not established here. A failed outer positive
control still prohibits using this procedure to declare the center shock-free.
Neither thresholds nor temperature alignment nor production output changed.
Raw input hashes were checked before and after analysis.

## Reproduce

```sh
python -m experiments.wind_bubble.check_outer_control_calibration
python -m unittest tests.experiments.test_outer_control_calibration tests.experiments.test_collision_front_motion
```

Use a fresh output directory. The default is
`outputs/outer_control_calibration_n128`, containing the figure, all tested
pairs, the eight comparison rows, known controls, and input/source fingerprints.
