# CPU instrumentation check — GPU confirmation still required

This is an end-to-end local CPU replay of the three-bubble 64-cubed setup,
with unchanged detector physics, thresholds, event schedule, and source budget.
It captures five selected physical states and evaluates the production masks.

The diagnostic checks succeeded: production mask decomposition and the
reconstructed exported surface indices/positions agree with the CPU replay.
The generated mask and profile figures were visually inspected at t=0.08.

**The strict comparison against the original GPU run failed.** The first
reported mismatch is in `surfaces_0001.npz:normals`, with a maximum absolute
difference among violations of approximately 4.06e-7. The comparison stops at
that mismatch; it does not establish that all later products agree. The
original rtol=1e-6 / atol=1e-8 checks were retained, and
`audit/verification.json` correctly records `passed: false`.

In this CPU replay, all 36 central-slice ROI cells fail alignment and pressure
jump at each of t=0.06, 0.08, 0.12, and 0.16. At t=0.08, 0.12, and 0.16 all
360 central-volume cells also fail both tests. Compression is present in some
cells, while the temperature-jump test passes. The combined zone is empty in
that volume at those three times, before surface selection or export.

These are preliminary CPU-replay findings, not a verified rejection report
for the original GPU movie and not a physical classification of the layer.
Run the prepared GPU audit and require its reference comparison to pass before
presenting these counts as an explanation of that movie.

The five `audit/state_*.npy` files total about 25 MiB and are local validation
inputs. They do not need to be included when committing the audit code for the
remote run. Run instructions and scope are in
`experiments/wind_bubble/COLLISION_DETECTION_AUDIT.md`.
