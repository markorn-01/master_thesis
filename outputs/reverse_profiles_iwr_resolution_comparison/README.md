# Reverse-sample inspection at 64³ and 128³

`reverse_sample_resolution_comparison.png` compares the preserved GPU
inspections at t=0.12, 0.15 and 0.20. Both use injection radius 0.0625,
21 requested events and dt_max=0.20/21. Backend and source-revision evidence
is documented in the individual result folders.

| Time | 64³ median RH residual | 128³ median RH residual |
|---|---:|---:|
| 0.12 | +49.79% | -5.82% |
| 0.15 | +36.04% | -7.55% |
| 0.20 | -2.82% | -8.66% |

Left: compare each direct density ratio with 4 M²/(M²+3), for gamma=5/3
and pressure-derived M. Curves show median per-sample fractional differences,
not errors against known true states. Shading spans the 16th–84th percentiles
of the surface-sample population. It is not a confidence interval.

Right: upstream radius minus injection radius. Curves show sample medians;
shading spans the minimum–maximum. Negative values lie inside injection.
At 64³, all 48 reverse upstream samples are inside at t=0.12; none are inside
at the other inspected times. At 128³, all endpoints are outside at all three
times. Samples are counted per detected surface cell, without area weighting.

The finer run removes early source overlap and reduces early density-jump
inconsistency. The later absolute median discrepancy increases, so these
results do not demonstrate monotonic improvement or convergence. Both the
numerical flow and the physical extent of the detector's cell-based sampling
change with resolution. The source radius and event schedule are controlled,
but these runs do not isolate reconstruction error from flow discretization
or source-proximity effects. Accurate thermalization cannot be inferred solely
from successful detection, online/offline agreement or a small median residual.

Reproduce the figure and summary table using NumPy and Matplotlib:

```bash
python outputs/reverse_profiles_iwr_resolution_comparison/plot_comparison.py
```

`comparison.csv` contains the plotted summary values. The generator reads
`local_checks.json` from each individual result folder; their verification
scripts describe provenance and readback checks. Individual profile plots and
all sampled thermodynamic pairs remain in those folders. The 128³ plot title
was corrected locally from saved CSVs, preserving the original image.
`SHA256SUMS` is a local preservation manifest.
