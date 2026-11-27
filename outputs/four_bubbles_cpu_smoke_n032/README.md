# Four-source CPU workflow smoke check

Completed locally on 27 November 2026, using JAX 0.6.2 on CPU.

- Four equal, noncoplanar tetrahedral sources.
- 32 cubed cells; three events at t=0, 0.01, 0.02.
- Prescribed per-source injection budget matched to the existing 64-cubed
  equal-two-source reference; 32-cubed mass-parameter scale 2.75.
- Shared online runner, optimized geometric tracker, stored-state offline
  comparison and whole-sequence batch-reference comparison enabled.

`equivalence.json` reports passing comparisons at all three events: 21 primary
history fields, zero maximum finite scalar difference, exact histograms,
matching sparse catalogs/background slices, eight group observations, exact
full tracking records, batch-reference agreement and final gas-state agreement.
`metadata.json` records the actual source/budget configuration and source hashes.

The run is an execution check, not collision or physical-ancestry validation.
It does not reach the planned 64-cubed pilot's interaction regime. These CPU
times are not a performance benchmark or general turbulent-box result.

The 64-cubed source preflight gives four masks of 160 cells each, shared mass
parameter scale 1.1, and prescribed per-source energy coefficient 0.32825707.
The intended remote run has 17 events through t=0.16. See
`experiments/wind_bubble/FOUR_BUBBLE_PILOT.md` for the commands and review scope.

Reproduce the smoke check in a fresh output directory:

```sh
python -m experiments.wind_bubble.run_four_bubbles \
  --num-cells 32 --num-analysis 3 --t-end 0.02 \
  --track-shocks --verify-offline \
  --output-dir outputs/four_bubbles_cpu_smoke_n032
```
