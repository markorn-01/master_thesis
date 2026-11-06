# First GPU two-bubble collision dataset

64³, 17 recorded events over t=0–0.16, two identical stationary EI sources at
(0.35,0.5,0.5) and (0.65,0.5,0.5), fixed injection radius 0.0625. User ran this
on compgpu9, physical GPU 5. Metadata records GPU backend, JAX 0.6.2,
Python 3.12.3 and revision 9b26d5a1d65f8b80d46481c90cd20c5d07a42b27.
No GPU model is inferred. All values use baseline dimensionless code units.

## Validation

- All eight focused two-source tests passed in the remote .venv-iwr environment.
- Remote online/offline comparison passed at all 17 events: 21 scalar history
  fields, maximum finite scalar difference zero, exact histogram counts,
  matching surface catalogs and background slices. Final gas states agree
  within relative tolerance 1e-6 and absolute tolerance 1e-8.
- verify_import.py checks the copied files without JAX. Saved online/offline
  CSVs and every NPZ array match exactly. Histograms, percentiles, source
  exclusion and per-source endpoint flags recompute correctly. Requested and
  actual times agree, and analysis source hashes match the local code.
- The original metadata includes one absolute remote source path. The readback
  script maps the known remote repository root to the local repository; it does
  not modify the original metadata or its hashes.
- Raw final gas states were not saved. Their agreement is the remote report,
  not an independent local check. This validates workflow consistency, not
  physical accuracy, complete shock coverage or resolution convergence.
- The full compatible remote regression suite remains outstanding.

## Inspected stages for clustering tests

| Event | Time | Surface samples | Visual interpretation |
| --- | --- | --- | --- |
| 0004 | 0.04 | 1328 | Separated bubbles |
| 0008 | 0.08 | 3096 | Near contact |
| 0016 | 0.16 | 6168 | Interacting bubbles |

Each event PNG shows Mach-colored refined surface positions and pressure/density
slices at z=0.5078125 (the cell-center plane just above the source midplane).
The final image was generated remotely; earlier views were generated locally
using the same viewer. All three were visually inspected. Scales are event-specific.
The near-contact frame does not establish an exact physical collision time.

Use these three events first for AstroLink position-only versus position-plus-normal
tests. Source positions can help interpret the separated geometry, but must not
be used to force cluster membership. Two sources do not imply exactly two shock
clusters: each bubble can contain more than one detected surface population.
After interaction, physical surface ancestry may be ambiguous. No cluster labels,
temporal correspondence or triangle meshes have been generated yet.

## Sampling and source interpretation

Minimum detected-surface clearance over the saved sequence is 0.1003181, or
6.4204 cells. This does not alone exclude all boundary influence.
At the final event all endpoint positions are known and no endpoints overlap
either source. Earlier events do have source overlap: downstream counts at
t=0.01, 0.02 and 0.03 are 616, 824 and 296, and upstream counts at t=0.12,
0.13, 0.14 and 0.15 are 120, 112, 80 and 48. These advisory flags do not filter
samples. Unknown counts and overlap must be assessed per event, not just at the end.

Each source occupies 176 injection cells. The existing analytic-volume source
normalization gives a discrete energy rate of 0.328257 per source versus the
nominal 0.5. Both sources have equal discrete rates. The 32³ trial used different
discrete rates, so these datasets are not an isolated shock-finder convergence
test. See the experiment guide for the unchanged source prescription.

## Reproduce and read back

Use a currently free GPU in the compatible environment, with a fresh directory:

```bash
python -m experiments.wind_bubble.run_two_bubbles \
  --num-cells 64 --num-analysis 17 --t-end 0.16 \
  --verify-offline --require-gpu \
  --output-dir outputs/two_bubbles_iwr_compgpu9_n064_repeat
```

Run local readback from the repository root:

```bash
python outputs/two_bubbles_iwr_compgpu9_n064/verify_import.py
```

local_checks.json records the readback result. SHA256SUMS preserves the copied
dataset and added inspection artifacts. The original online outputs remain unchanged.
