# Reverse-candidate radial inspection

This diagnostic reruns the hydrodynamic single-bubble problem at 64³ with the
same injection radius (0.0625), 21 requested output times over 0–0.20 and
maximum timestep (0.20/21) as the temporal demonstration. It retains snapshots
in memory for inspection and analyzes indices 12, 15 and 20. It does not change
the solver, shock detector, Mach estimator, or tracking rules.

After syncing the code and activating `.venv-iwr`, run on an available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m experiments.wind_bubble.check_reverse_profiles \
  --num-cells 64 --require-gpu \
  --output-dir outputs/reverse_profiles_iwr_compgpu3_n064
```

Use a new output folder if repeating the experiment. Omit `--require-gpu` for
a separate CPU demonstration. Different backends are separate evidence;
matching configuration alone does not establish identical gas states.

## Saved products

- `reverse_profiles.png`: density, pressure, radial velocity and radial flow
  Mach profiles at three times, using spherical shell medians of width dx.
- `profile_0012.csv`, `profile_0015.csv`, `profile_0020.csv`: plotted shell medians.
- `samples_*.csv`: all detected samples outside injection, radial band labels,
  normal alignment, profile-aware Mach, selected upstream/downstream radii,
  densities, pressures, and numerical sampling flags. Sampling positions
  use the actual normal and selected distances, not a radial approximation.
- `sample_checks.csv`: reverse-sample counts, pressure/density compression
  counts and numbers of selected endpoints inside the injection region.
- `metadata.json`: backend, schedule, revision, source hashes, and completion status.

No full gas arrays are saved. This is a targeted validation workflow, not a
performance benchmark or a replacement for the online runner.

## Reading the figure

Dashed lines show median radii of the two detected radial populations. The
shaded region is the prescribed injection region. Colored vertical sample
lines and thermodynamic markers correspond to one valid reverse sample,
chosen deterministically as the point nearest the reverse population's median
radius. These are individual interpolated samples; they need not lie on the
spherical median curves. The complete sample table supports population checks
rather than relying only on this illustrative point.

The last row distinguishes radial flow Mach, |v_r|/c_s in the simulation frame,
from pressure-jump shock Mach. They are different quantities. The latter uses
pressure contrast to infer upstream Mach in the shock frame; it is not an
independent measurement of propagation speed.

A pressure increase is partly enforced by the sampler and used to infer Mach;
its presence therefore is not independent validation. Density compression,
sample locations relative to the source, inward normals, and the surrounding
profiles add complementary evidence. Even these do not by themselves establish
convergence, accurate plateau recovery, or Rankine–Hugoniot consistency.
Endpoints within the injection region are flagged without changing the measured
values or discarding detections. Higher resolution may be needed if source-region
sampling or a broadened transition dominates the inferred jump.

## First local result (CPU)

The completed 64³ CPU inspection is preserved in
[`outputs/reverse_profiles_cpu_n064_retry/README.md`](../../outputs/reverse_profiles_cpu_n064_retry/README.md).
All 48 upstream endpoints at t=0.12 lie inside injection, although the surface
cells lie outside. None of the upstream/downstream endpoints at t=0.15 or 0.20
lie inside. The sampled density contrast also departs substantially from the
pressure-Mach RH prediction at the first two times. This motivates checking
source proximity and resolution before interpreting early reverse-shock power.
Repeat on the GPU before attributing these findings to the preserved GPU run.
Two focused geometry tests and independent NumPy readback checks passed.

## GPU confirmation

The corresponding 64³ GPU inspection on compgpu3, physical GPU 0, also
completed. Imported source hashes, times and sample counts passed independent
readback checks. It confirms 48/48 upstream endpoints inside injection at
t=0.12, with no inside-source endpoints at t=0.15 or 0.20. Median fractional
density-jump residuals relative to the pressure-Mach RH prediction are +49.8%,
+36.0% and -2.8%, respectively. These are diagnostic discrepancies, not errors
relative to known exact states. Avoid treating early reverse-candidate power
or cumulative energy as quantitatively validated solely by online/offline agreement.
See [`outputs/reverse_profiles_iwr_compgpu3_n064/README.md`](../../outputs/reverse_profiles_iwr_compgpu3_n064/README.md)
for the figure, checks and interpretation limits. No physics or filtering was changed.

## 128³ comparison

The 128³ GPU inspection also passed independent readback checks. All endpoints
are outside injection at the three inspected times. Median density-RH residuals
are -5.82%, -7.55% and -8.66%; the early discrepancy improves relative to 64³,
but the late absolute median discrepancy does not. This is not a convergence
claim. See the [comparison figure and interpretation](../../outputs/reverse_profiles_iwr_resolution_comparison/README.md)
and [128³ evidence](../../outputs/reverse_profiles_iwr_compgpu3_n128/README.md).

The plot title now uses the supplied resolution. The originally imported 128³
image had a 64³ title and is preserved alongside a corrected CSV-based replot.
Only plotting changed. Three focused geometry/title tests passed locally.
