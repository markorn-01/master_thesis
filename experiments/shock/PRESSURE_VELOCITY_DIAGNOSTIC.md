# Pressure and velocity diagnostic for rejected shock candidates

This experiment asks whether pressure and converging velocity can locate a
shock when a nearby contact reverses the measured temperature direction.
The answer is useful but limited: these signals recover candidate locations
in the known missed-shock references, and also flag smooth compression.
The additional full-state checks fail the outer bubble shock control, so
this diagnostic cannot settle whether the central bubble layer contains
missed shocks. It is a separate experiment, not a production correction.

## What the diagnostic does

1. Use the pressure gradient to choose a direction, without a temperature
   alignment test. Select local compression peaks along that direction.
2. Sample both sides at distances of 0.5, 1.5, 2.5, and 3.5 cell widths.
   Compare the measured velocity drop with the drop predicted by the
   pressure jump for an ideal-gas shock.
3. Check whether pressure and normal velocity change little when either
   endpoint is moved another half cell. Reject windows that cross an
   opposing pressure direction. Bubble windows must also stay within the
   domain and avoid every injection sphere along the whole segment.
4. Separately compare the sampled density jump, entropy increase, momentum,
   enthalpy, and tangential velocity with the same shock relation.

“PV” means the pressure/velocity and endpoint checks pass. “Full” additionally
means the sampled states meet the stated jump-consistency checks. Neither
label establishes shock identity, temporal persistence, or a physically
resolved discontinuity. In particular, shock speed inferred from mass balance
is not an independent measurement of propagation.

The minimum pressure-inferred Mach number is 1.3. Jump and endpoint residuals
are reported at 5%, 10%, and 20%; entropy-signal error must be at most 50%.
These are declared experimental settings, not universal accuracy thresholds.
The results below use 10% unless specified otherwise. Candidate counts refer
to sample locations, not physical shock objects.

The separation of candidate selection from state-jump checks follows the
general approach used in [SHOCKFIND](https://academic.oup.com/mnras/article/463/1/1026/2589921)
and the consistency checks discussed by
[Schaal et al.](https://arxiv.org/html/1604.07401). This implementation is not
a reproduction of either finder: its pressure normal, finite windows, and
residual thresholds are defined here.

## Reference results

The inputs are the frozen 228 cases from [the shock/contact study](SHOCK_CONTACT.md),
plus six negative controls. No new hydrodynamic simulation is evolved.

| Input | Result at 10% |
| --- | --- |
| 12 isolated shock controls | All have a correctly located PV and full-compatible sample |
| 36 moving contacts alone | No PV candidates |
| Six shock/contact cases previously rejected at the zone stage | All have a correctly located PV candidate; none has a full-compatible state pair |
| Four smooth isentropic compressions | All produce PV candidates; none passes full-state checks |
| Stationary pressure bump and uniform-pressure compression | Neither produces PV candidates |

The smooth controls have constant `P/rho^gamma`: they are smooth compressive
initial profiles, not entropy-producing shock discontinuities. They show why
accepting pressure/velocity candidates directly would introduce false
detections. Conversely, the unresolved reference really contains a physical
shock, but its grid samples mix the contact and shock states. Failing the
full-state test therefore does not prove absence of a physical shock.

The four examples are plotted in
[diagnostic_controls.png](../../outputs/pressure_velocity_reference_diagnostic/diagnostic_controls.png).
The complete 180-pair matrix has correctly located PV evidence in 176 cases
and full-compatible evidence in 74. These constructed-case counts are not
an astrophysical detection rate.

## Saved three-bubble results and the failed positive control

The unchanged diagnostic was applied to ten saved states: five times
(`0.04, 0.06, 0.08, 0.12, 0.16`) at each of 64³ and 128³. Original saved-state
hashes were checked before analysis. Full-grid gradients are used before
selecting candidates within either spatial box.

The central box is `x=[0.45,0.55], y=[0.335,0.415], z=[0.42,0.58]`.
The outer-forward-shock control is
`x=[0.05,0.3125], y=[0.335,0.415], z=[0.46,0.54]`.

- At the later central times (`0.08, 0.12, 0.16`), no PV or full-compatible
  sample passes at 10%. Eight geometric candidates at 64³, time 0.08 have
  no source-free, in-domain sampling window; their absence of evidence is
  recorded explicitly.
- Across the ten outer-control observations, 144 PV locations pass, but
  **no full-compatible location passes**. Even the 20% full-state threshold
  accepts none in this positive control.

This control failure prevents using the central rejection to rule out
physical shocks. The full-state diagnostic has not demonstrated sufficient
sensitivity to the evolved, curved bubble shock with these sampling windows.
The [central overlay](../../outputs/three_bubble_pressure_velocity_diagnostic_n064_n128/pressure_velocity_overlay.png)
shows candidate geometry, not certified shocks. Its counts cover the 3D box;
displayed symbols are restricted to within 0.75 cell of the plotted slice.

A bounded sensitivity probe tried upstream and downstream distances
independently on four outer rays nearest the source axis at 128³, time 0.16.
Of 64 distance pairs, even the best has a largest residual of about 27.6%,
dominated by density-jump disagreement. This probe neither exhausts possible
state sampling nor validates those rays independently. It did not justify
relaxing the consistency thresholds.

## Decision and remaining question

Keep this diagnostic separate from the production finder. It supplies useful
locations for inspection, but has not supplied reliable replacement shock
states or thermalization measurements. No surface catalog, Mach statistic,
group identity, or heating output is changed by this experiment.

The reference study has established that an unresolved shock/contact pair
can be missed. Whether that mechanism explains the bubble's central layer
remains open. The next physical test should resolve and follow the actual
collision layer more closely, while matching source budgets and event times
and retaining the outer-front positive control. Acceptance must require
consistent state sampling and propagation evidence; a higher grid size or
a larger candidate count alone is not success.

## Reproduction and validation

Local validation uses Python 3.10, JAX 0.6.2, and CPU execution. GPU validation
of these new diagnostics is outstanding. The complete shock-finder suite
passed: 79 tests in 34.603 seconds, including eight new tests. They cover exact
jump relations, unresolved and resolved shock/contact pairs, contact and
smooth-compression controls, Galilean boosts, 3D extrusion, input immutability,
and opposing-direction sampling rejection.

```bash
python -m unittest discover -s tests/physics_modules/shock_finder -p 'test_*.py'

python -m experiments.shock.check_pressure_velocity_diagnostic \
  --output-dir outputs/pressure_velocity_reference_new

python -m experiments.wind_bubble.check_collision_pressure_velocity \
  --output-dir outputs/three_bubble_pressure_velocity_new
```

Use fresh output directories. The reference runner reads the existing frozen
shock/contact profiles. The bubble runner defaults to the existing verified
64³ and 128³ collision-audit folders; `--roots` accepts alternative folders
with the same saved-state format. Verification files fingerprint all direct
input files and selected analysis sources. CSV files preserve per-window
residuals, acceptance flags, and sampling exclusions for inspection.
