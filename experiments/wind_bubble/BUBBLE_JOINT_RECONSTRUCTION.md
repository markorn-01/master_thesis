# Transfer of joint reconstruction to the saved bubble profiles

## Decision

**The fixed planar reconstruction is not usable on the tested bubble profiles.
Do not promote it to production or use it to certify the missing central fronts.**

The preceding [controlled feasibility test](../shock/JOINT_SHOCK_CONTACT_FIT.md)
showed that a hidden shocked state can be inferred *when the assumed planar fan
and blur model is correct*. This separate check tests transfer to existing
simulation data. It changes neither that calibration nor the production finder.

## Bounded protocol

Input: the completed, fingerprinted 128-cubed run in
`outputs/collision_front_motion_iwr_n128`, at times 0.04, 0.06, 0.08, 0.12, 0.16.

Four native rays are used: the two grid coordinates closest to the lower-pair
axis in each transverse direction. They lie at
`y = 0.37109375, 0.37890625` and `z = 0.49609375, 0.50390625`.
No transverse interpolation, upsampling, padding, or profile averaging is used.
The one-dimensional fit receives density, x velocity, pressure, and grid spacing.
Other velocity components are retained in the saved window data, and their
magnitude relative to sound speed is reported; a full three-dimensional fan is
not being fitted.

At each time and ray, select the outermost pressure-gradient feature on each
outer side as a known shock control. Select the pressure-gradient feature
closest to the collision midplane on each central side for the original question.
Selection uses pressure and geometry, not fit quality. The saved production
surface flag is checked within two cells as context, without recomputing or
modifying the historical masks.

Use one 16-cell window centered on the selected feature and clamped within its
admissible connected interval. Entire grid-cell cubes intersecting any injection
sphere are excluded. Central windows must stay on their own side of the collision
midplane. These guards prevent treating prescribed source gas or both opposing
fronts as one shock/contact fan. No window search is performed to improve fits.

The fitter source fingerprint must match the successful calibration. All fit
settings, blur hypotheses, restarts, competing models, and the 0.001 maximum
normalized-residual requirement are unchanged. A valid shock-only model can
supply an outer control without requiring an invented contact. A joint model
must satisfy all its original identifiability and stability gates.

## Results

| Check | Cases | Outcome |
|---|---:|---|
| Outer control windows: 5 times × 4 rays × 2 sides | 40 | All models rejected as mismatches |
| Nearby saved production surface at those outer controls | 40 | Present in every case |
| Central candidates: 5 times × 4 rays × 2 sides | 40 | Eight admissible cells each; not fitted |

For the outer controls, the best joint models had maximum scaled residuals
from about **0.148 to 0.235**, versus the fixed acceptance limit **0.001**.
No competing shock-only, contact-only, or isentropic model supplied an acceptable
fit either. These residuals are divided by the fitter's data-derived conserved
scales; they are not direct relative errors in every physical variable.

The central interval between each lower source boundary and the midplane has
only eight admissible cells at this resolution. The current prototype requires
16 observations, so these windows are explicitly unavailable. This minimum is
an implementation and validation requirement, **not a theorem that eight cells
can never contain shock information**. Reducing it would constitute another
method change requiring separate validation. Crossing a source or joining
opposing fronts would instead change the physical model being tested.

The figure shows rejected candidate fits at t=0.12. Candidate parameters remain
in `fits.jsonl` for inspection, but no accepted Mach number or shock position is
reported for failed cases.

## Interpretation and stopping point

The model assumes constant states separated by one planar shock and one contact,
with a small, prescribed family of conserved-variable blur operators. It cannot
match these evolving bubble controls under those assumptions and tolerances.
Curvature, spatial gradients, numerical broadening, and additional waves are
possible reasons, but this check does not isolate their individual contributions.

This closes the transfer test with a negative result. The analytic success does
not justify replacing the production finder, and the rejected bubble fits do
not establish physical shock absence. Do not relax the fit tolerance until an
apparently acceptable central answer appears.

The defensible scientific findings are distinct:

- Known neighboring fronts should be recoverable when resolved; the corrected
  opposing-front benchmark demonstrates recovery in the tested exact cases.
- A nearby contact can conceal a known shock from the present local tests and
  corrupt the sampled immediate downstream density.
- A restricted joint reconstruction can recover that hidden state in controlled
  inputs, but this particular implementation fails the real bubble controls.
- Whether the specific late central layer contains unresolved physical shock
  fronts remains unverified. Neither grouping nor surface smoothing can answer
  that hydrodynamic question or repair missing input detections.

A stronger claim about that layer requires a separately validated method for
the evolving multiwave flow or independent, resolved propagation/jump evidence.
It is a separate research task, not a missing plotting setting. The current
checkpoint should be used to state the finder’s demonstrated limitation, rather
than promising that another parameter change will settle the physical question.

## Reproduce and inspect

```bash
MPLCONFIGDIR=/private/tmp/master-thesis-mpl-benchmark \
python -m experiments.wind_bubble.check_bubble_joint_reconstruction \
  --output-dir outputs/bubble_joint_reconstruction_n128
```

Use a fresh output directory. No GPU or simulation rerun is required.

The result folder contains `summary.csv`, full candidate fits in `fits.jsonl`,
the native fitted windows in `fitted_profiles.npz`, `bubble_transfer.png`, and
`verification.json`. Sources and original inputs are fingerprinted and checked
again after execution. The pilot run is preserved separately in `/private/tmp`;
only the finalized protocol result is retained in the repository.
