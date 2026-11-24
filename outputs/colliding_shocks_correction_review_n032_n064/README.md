# Opposing-front correction review

`before_after.csv` compares all 96 cases to the frozen pre-correction run.
Six incoming/outgoing pair cases newly meet the unchanged location-and-5%-Mach
criterion; no previously passing pair case fails it after correction.
`corrected_sampling_trace.csv` checks actual production sampler endpoints
against the saved 3D Mach results for both exact two-cell cases at 64^3.
`corrected_close_fronts.png` illustrates those cases.

All four exact two-cell cases at 32^3/64^3 pass. Isolated/wide exact controls
remain correct; contact controls have no false surface. This does not imply
completeness for sub-cell or heavily broadened shocks.

`verification.json` records input/source hashes and comparison outcomes.
`validation.json` records local checks. The distinct bubble recheck is in
`../three_bubble_overlap_fix_recheck_n064_n128/`. Full explanation:
`../../experiments/shock/OPPOSING_FRONT_CORRECTION.md`.
