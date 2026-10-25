# First online/offline single-bubble comparison

The online run completed on local CPU with a 32³ grid, physical injection radius
0.0625 and analysis times 0, 0.05, 0.10, 0.15 and 0.20. Shock analysis ran during
integration. The online path retained no complete gas snapshot series and wrote
only history rows and per-event histograms, followed by a summary plot.

A separate offline reference run retained snapshots at the same times. Its host
analyzer produced matching compact diagnostics:

- All **41 history fields** agreed at all five events within rtol=1e-6 and
  atol=1e-8, including matching missing-value locations.
- Histogram counts matched exactly.
- Maximum absolute scalar difference was approximately 3.87e-7.
- The final gas states agreed within the same stated tolerance.
- Initial uniform gas had no detected surface; final detection had 872 surface
  cells outside the injection region.

All **89 local regression tests passed** in 123.189 seconds, including the new
compact-product tests and the existing analyzer and timing tests. A separate
three-event online-only smoke run to t=0.02 also completed without allocating
an offline reference series. GPU validation remains outstanding.

This is a low-resolution correctness demonstration, not convergence evidence.
The new density-jump diagnostic frequently gives values below one at this
resolution. It reports sampled densities at pressure-selected positions; finite
positive samples are not a guarantee of a physically resolved compression jump.
No estimator adjustment was made to force a compressive ratio.

The online callback still transfers dense diagnostic grids to the CPU, where
statistics are reduced. No full gas states or dense diagnostic grids are saved.
Runtime and peak-memory overhead have not been benchmarked. The offline
reference intentionally stores snapshots only for verification.

`equivalence.json` records comparison tolerances and results. `metadata.json`
records the environment and setup. The `online/` and `offline_reference/`
directories contain corresponding compact products. The figure is
`online_histories.png`.

See `experiments/wind_bubble/ONLINE_ANALYSIS.md` for the implementation contract,
limitations, tests and remote validation command. No commit or push was made.
