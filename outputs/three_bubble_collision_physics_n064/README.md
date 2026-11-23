# Independent physical assessment of the lower collision layer

This analyzes the five captured GPU states at t=0.04, 0.06, 0.08, 0.12 and
0.16 from the verified 64^3 three-bubble audit. No solver, detector, source,
normal, Mach estimator, clustering or rendering code was changed. No new
simulation was needed for this assessment.

## Findings established from the saved states

The later central dense ridge cannot be connected to its hot surroundings
as a single resolved compressive Euler shock using the inspected state pairs.
Density rises inward, while temperature and the entropy proxy K=P/rho^gamma
fall. A proposed compressive shock connecting those states would have a
negative specific internal-energy change and a decreasing specific entropy.
That is incompatible with the single-shock jump/entropy conditions.

This is an independent thermodynamic inconsistency, not a repetition of the
production gradient-alignment test. It does not assert that a gas parcel's
entropy decreases in time: the profiles compare different spatial locations.

For the four grid lines nearest the lower-source axis, every possible pair
of sampled centers on each side of the layer was checked separately:

| Time | Source-free pairs | Higher P and density | Also converging and entropy-increasing | Full necessary shock conditions |
| --- | --- | --- | --- | --- |
| 0.04 | 48 | 16 | 8 | 0 |
| 0.06 | 48 | 48 | 0 | 0 |
| 0.08 | 48 | 48 | 0 | 0 |
| 0.12 | 48 | 0 | 0 | 0 |
| 0.16 | 48 | 48 | 0 | 0 |

The x-directed sampling corridor is x=0.421875–0.578125, split at x=0.5 so opposite flows
are never treated as the two states of one shock. The four lines use y in
{0.3671875,0.3828125} and z in {0.4921875,0.5078125}. All segments intersecting
any source sphere are excluded, not just those with an endpoint inside it.

The wider 60-line check spans y=0.335–0.415 and z=0.42–0.58, with 1104
source-free pairs per time. At t=0.08 all 1104 are pressure/density-increasing
but none has the required entropy increase. At t=0.12 and 0.16 there are
174 and 1096 pressure/density-increasing pairs, respectively, again with
no entropy-increasing pairs.

## Independent normals and conservation checks

The pressure-gradient check uses all 360 centers in the earlier audit box,
normals along +grad(P), and half-widths of 1, 2 and 3 cells. It samples the
primitive states by trilinear interpolation, without using a production
shock mask or the temperature normal. Some off-axis pairs do pass the
entropy test; these were retained for further testing, not hidden.

Across these pressure-directed samples, none satisfies the combined
compressive, entropy and shock-frame sonic ordering conditions together
with momentum, total-enthalpy and tangential-velocity residuals below 5%,
10% or 20%. Residual limits are diagnostic sensitivity levels, not a new
production detector or a proof of shock absence.

The proposed front speed is inferred from mass conservation for each pair.
Its mass residual is therefore zero by construction and supplies no
independent confirmation. It is not a speed measured by temporal tracking.
The time-position figure shows the gas structure's evolution without
labeling a density/pressure ridge as a physical shock trajectory.

## Why these results do not prove that no physical shock exists

The same checks were also applied to an outer forward front as a positive
comparison. At this coarse resolution, no inspected outer-front pair passes
the full 5% or 10% compatibility checks either. At 20%, some outer pairs
pass at t=0.12 and 0.16, but none at the earlier three times. This demonstrates
that strict isolated-shock endpoint tests are not reliable absence tests on
these broadened, spatially varying grid states.

Consequently, the evidence supports the statement:

> The visible dense ridge is not established as one resolved shock. Its
> bulk thermodynamic contrasts are inconsistent with such an interpretation.
> A missed embedded physical front is neither proved nor ruled out at 64^3.

The earlier result remains firm: alignment and directed pressure-jump
criteria cause the production detection gap before grouping or smoothing.
This assessment gives no justification for weakening those criteria merely
to mark the dense ridge as a shock.

## Evidence and validation

- [Profiles at all five captured times](collision_entropy_profiles.png).
- [Time-position views from all 17 saved slices](collision_time_position.png).
- `axis_state_pairs.csv`: every axial endpoint pair, including the outer
  comparison and all residuals, rather than a selected best match.
- `pressure_normal_state_pairs.csv`: every valid pressure-directed pair.
- `summary.csv` and `assessment.json`: counts, input/source hashes and scope.

All input files were checked against their hashes before and after reading.
Exact weak/strong shocks, an oblique moving shock, an advected contact,
cold dense compression, and smooth isentropic compression provide local
controls. The focused physical, resolution-plan and existing audit suites
passed 20 tests. Both figures were visually inspected.

Theory references: the [Clawpack Euler/Riemann chapter](https://www.clawpack.org/riemann_book/html/Euler.html)
for conservation, entropy and shock/contact distinctions; and
[Schaal & Springel (2015), Sections 2.2–2.3](https://arxiv.org/html/1407.4117)
for the shock-finding criteria and effects of numerical broadening. These
references supply physical consistency conditions, not a classification of
this particular simulation.

## One bounded follow-up

The prepared 128^3 run preserves physical source positions, injection radius,
per-source prescribed mass/energy rates, output times and solver timestep cap.
It captures the same five events and runs these same physical checks. Sampling
uses both fixed-cell widths and physically matched widths: {1,2,3} cells at
64^3 and {1,2,3,4,6} at 128^3. Axial/outer-front windows retain fixed physical
extents. A two-resolution comparison is not a convergence proof.

If distinct thermodynamically admissible fronts become resolved, compare
their locations with the original detector masks to test actual missed
detections. If clean states remain unavailable, report the unresolved
physical classification explicitly; do not infer absence or repeatedly
relax jump tests until something passes.
