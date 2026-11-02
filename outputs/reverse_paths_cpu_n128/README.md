# 128³ reverse-candidate normal paths at t=0.20 (CPU)

Completed locally on 2026-11-02 with Python 3.12.4, JAX 0.6.2 and CPU backend.
This is a new integration of the 21-event baseline setup with injection radius
0.0625 and dt_max=0.20/21, not a reconstruction of saved GPU gas states.
Metadata source hashes identify the actual working-tree implementation.

All 2624 valid reverse samples were inspected. Independently interpolating the
selected endpoints reproduces the analyzer's pressure/density values (maximum
absolute difference 8.94e-8). Reconstructing the original six pressure candidates
also reproduces both selected distances. These checks found no coordinate,
sign or pressure-choice mismatch on these paths. They do not prove accurate
upstream/downstream reconstruction or establish absence of every possible bug.

All selected endpoints are outside the Boolean shock zone and injection region.
However, one-cell neighborhoods around upstream endpoints often extend back into
the Boolean zone. The downstream pressure and density neighborhoods are flatter
than a broadened transition, but not constant. The original endpoint-to-endpoint
median changes are about 13.4%/2.95% in upstream/downstream pressure and
11.6%/2.47% in density. The full three-point range additionally captures curvature;
its population results are in the companion comparison.

Every downstream pressure selection uses the farthest candidate, 1.5 cells
beyond its zone exit. Holding the selected upstream state fixed and hypothetically
using each of the six downstream candidates produces median density-RH residuals
from approximately -11.0% to -8.66%. These hypothetical values are a sensitivity
check only; no measured values are replaced, and no estimator is tuned to fit
RH. See `normal_path.png` for the median-residual illustrative sample and
`normal_paths.npz` for all sampled paths and candidate states.

Independent NumPy readback checks passed, including source hashes, times,
endpoint values/radii, zone flags, indices and the defining pressure-Mach relation.
Full gas arrays were not saved and cannot be independently rechecked. The
comparison does not claim CPU/GPU equality or a live GPU validation of full paths.

See [`../reverse_path_diagnostics/README.md`](../reverse_path_diagnostics/README.md)
for interpretation, plots, definitions, limitations and reproduction commands.
