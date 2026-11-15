# Completed GPU extension to t=0.20

128³, two unequal wind sources (1:0.5), 21 analysis events. Original first
17 requested times and dt_max=0.16/17 were preserved. Seven full states were
captured at events 10,12,14,15,16,18,20. The five original captured state hashes
are unchanged. GPU audit passed; local readback additionally establishes exact
agreement of 680 NPZ fields across the original 17 events, not merely agreement
within tolerance (`copy_checks.json`).

All seven full states are finite and match recorded hashes. Fourteen source
crops match the full states and production criterion intersections. All 56
source/region count rows agree. Source hashes match. Minimum detected-surface
boundary clearance: 0.06799094 domain units = 8.70284 cells. This does not
establish that all gas disturbances/endpoints are unaffected by boundaries.

At weaker-source local inward cells: event 18 has 248 zones/surface/exported
samples; event 20 has 512 zone cells, 488 selected/exported surface cells.
No production thresholds or source prescriptions changed.

`audit/state_*.npy` occupies 293,602,176 bytes. Files remain on disk but are
excluded from Git by `audit/.gitignore`. Preserve original GPU/local files;
full-state hashes are recorded. The compact catalogs, source crops, counts,
verification and local copy-check report can be versioned.

Analysis and interpretation: `outputs/two_bubbles_detection_extension_review_n128/`.
Reproduce in a new directory using `experiments/wind_bubble/DETECTION_EXTENSION.md`.
