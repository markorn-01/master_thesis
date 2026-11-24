# Corrected finder: online/offline integration smoke check

Completed locally on CPU/JAX 0.6.2 using commit 02dc807. Two bubbles, 32^3,
five analysis events over t = 0 to 0.16, with incremental grouping/tracking.

Run command:

```bash
python -m experiments.wind_bubble.run_two_bubbles \
  --num-cells 32 --num-analysis 5 --t-end 0.16 \
  --track-shocks --verify-offline \
  --output-dir outputs/opposing_front_online_smoke_cpu_n032
```

`equivalence.json` records online/offline primary-product agreement, sparse
surfaces, background slices, full tracking-record agreement, batch-reference
agreement, and final gas-state agreement. The simulation is repeated for
validation: one online run and one retaining states for offline analysis.

This tests consistency of the corrected workflow. It does not establish
physical accuracy, resolve the central bubble gap, or validate the GPU backend.
The 32^3 grid and sparse cadence are intentionally a small integration check.
