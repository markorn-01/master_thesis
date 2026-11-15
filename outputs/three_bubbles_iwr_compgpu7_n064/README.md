Completed three-source GPU run on compgpu7, slot 2, JAX 0.6.2: 64³,
17 analysis events through 0.16. Three equal-strength non-collinear sources,
160 injection cells each, each prescribed energy coefficient ~0.32825707.
Total prescribed coefficients are 1.5 times the two-source equal reference;
this is a new geometry, not an isolated convergence test.

GPU online/offline verification passed, including final gas-state comparison.
Local readback independently matches all 21 scalar fields at every event and
680 NPZ fields exactly. Source hashes match. Actual/requested times differ by
at most 1.49011612e-7, within the original runner tolerance of 1e-6; recorded
actual times are used in tracking. These times are not claimed to be exact
floating-point requested timestamps.

Final catalog: 7,134 surface samples, 42 upstream source overlaps, zero downstream
overlaps and zero unknown endpoints. Minimum detected-point boundary clearance
across saved events is 0.12727273 = 8.14545 cells. This does not establish that
all gas disturbances and profile endpoints are unaffected by boundaries.

`event_0016.png` shows final points colored by Mach and central background slices.
Hollow slice stars are projected sources outside that slice, not missing winds.
`copy_checks.json` records the local copy validation. All data remain unlabeled
with respect to physical shock/source ownership.
