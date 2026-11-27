# Unequal four-source CPU smoke check

Completed locally with JAX 0.6.2 on CPU: 32³ cells, three events over
`0 ≤ t ≤ 0.02`, strengths `[1.5, 1, 0.75, 0.75]`, online tracking and
stored-state verification enabled. Positions and injection radius match the
equal four-source pilot; all sources are active from the beginning.

All three events passed the 21-field comparison (maximum finite scalar
difference zero), exact histograms, sparse catalogs, slices, full tracking
records, independent batch tracking and final-state comparison. Eight group
observations were compared. See `equivalence.json` and `metadata.json`.

The strengths sum to four, preserving the equal four-source pilot's total
prescribed mass and mechanical-energy coefficients. The prescribed energy
rates are approximately `[0.492386, 0.328257, 0.246193, 0.246193]`. They are
coefficients before the state-dependent EI conversion, not measured net
gas-energy deposition.

This is an early-time execution check, not collision validation or a runtime
benchmark. The 64³ interaction run remains to be performed on a GPU. Its
protocol is in `experiments/wind_bubble/FOUR_BUBBLE_UNEQUAL_PILOT.md`.
