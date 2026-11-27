# Unequal four-source GPU run: copied and checked

Run by the user on compgpu2 and copied locally. Complete 64³ sequence with
17 events from t=0 to t=0.16, strengths `[1.5,1,0.75,0.75]`, the fixed
tetrahedral source layout, online tracking and offline verification.

All 180 expected original files are present and readable, with no extra
simulation files or nested copies. All 25 recorded source fingerprints match
the local checkout. Local readback repeated the 21-field online/offline
comparison (zero maximum finite scalar difference), exact histogram counts,
sparse catalogs, slices, full tracking products and the independent batch
reference. All passed. The GPU's final gas-state comparison passed; full gas
states were not copied, so this particular check cannot be repeated locally.

Tracking records contain 71 group observations and 25 continuation segments;
these are geometric records, not physical shock counts. At the final event,
10,187 samples include retained groups of 9,005, 553, 333, 127 and 127 samples,
plus 42 samples excluded by grouping. The minimum detected-surface boundary
clearance is 0.10466 domain units, about 6.70 cells.

No sampling endpoints are unknown. Downstream source overlaps occur at early
times, and upstream overlaps affect 44 samples at t=0.14 and 46 at both
t=0.15 and t=0.16. These advisory flags remain in the statistics.

See `copy_checks.json` for verification and original file hashes. The completed
equal/unequal comparison, figures and interpretation are in
`outputs/four_bubbles_equal_unequal_comparison_n064/README.md`.
