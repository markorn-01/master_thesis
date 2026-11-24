# Corrected finder: known collision matrix

Same 96 analytic-state and broadening-control cases as the frozen baseline
`../colliding_shocks_reference_n032_n064/`, now using opposing-normal guards
in surface selection and sampling. Run on CPU/JAX 0.6.2.

`summary.csv`, `front_metrics.csv`, and `profiles/` retain the original
measurement definitions. `verification.json` records the actual corrected
source hashes; `source_files_unchanged_during_run` means sources did not change
while the benchmark ran, not that this is the pre-correction implementation.

All four exact two-cell regressions (incoming/outgoing, 32^3/64^3) now recover
both fronts with correct Mach. Broadening/sub-cell limitations remain.
See `../colliding_shocks_correction_review_n032_n064/` for the explicit
before/after comparison and corrected endpoint traces.
