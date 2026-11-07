# Budget-matched two-bubble GPU run, 128³

Imported from the user-run compgpu3 experiment on physical GPU 0. Metadata records
JAX 0.6.2, GPU backend, completion, and 17 events over t=0–.16. The user reported
15 focused remote tests passing before execution.

The mass-rate parameter is .00791011235955 per source. With 1780 injection cells
per source, the discrete mass rate is .00656514140 and the prescribed mechanical-
energy coefficient .32825707013, matching the 64³ reference. These coefficients
do not establish equal cumulative net gas-energy deposition under the existing
state-dependent pressure update.

`metadata.json` and `online/` are unchanged imported solver products.
`local_checks.json` records the subsequent saved-product validation. No full
gas-state sequence or offline reference was retained at this resolution.

Paired figures and analysis are in
`../two_bubbles_resolution_comparison_n064_n128/`; see the
[comparison report](../../experiments/wind_bubble/TWO_BUBBLE_RESOLUTION_RESULTS.md).
