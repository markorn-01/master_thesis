# Bubble recheck after opposing-front corrections

Old and corrected detection use the same ten saved gas states at 64^3/128^3,
with verified original state hashes. Historical modules come from the pinned
Git revision recorded in `verification.json`; both are evaluated on CPU.
No integration, tracking, or smoothing is run.

`central_stage_counts.csv` shows identical central zone/surface counts. The
later central volume has zero candidates and surfaces at t = 0.08, 0.12, 0.16
at both resolutions, before and after correction. The gap is still caused
earlier than the stages changed here. This does not establish that the
rejected region contains no physical shock.

`archived_backend_comparison.csv` additionally confirms that the historical
CPU zone/surface crop masks agree with the archived GPU masks in all ten
cases. `bubble_before_after.png` visualizes selected central slices;
`verification.json` records the complete provenance and scope.
