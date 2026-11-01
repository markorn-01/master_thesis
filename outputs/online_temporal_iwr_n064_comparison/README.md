# Reverse-candidate comparison at 64³

`reverse_candidate_cadence_comparison.png` (and PDF) compares preserved online
outputs from compgpu3 GPU 0 at 5 and 21 analysis events. Both online/offline
comparisons passed, including independent local readback and power integration.
Reproduce the figure with NumPy and Matplotlib:

```bash
python outputs/online_temporal_iwr_n064_comparison/plot_comparison.py
```

## Reading the figure

- Radius: the finer schedule detects the inner candidate at t=0.12 instead
  of t=0.15 and provides nine detections instead of two. The final radii are
  approximately 0.099797 in both runs. The dotted line is the injection radius.
- Normal alignment: values near -1 indicate normals pointing inward, consistent
  with the reverse-candidate interpretation. This does not prove accurate Mach
  numbers or fully resolved upstream/downstream sampling.
- Power: the estimated surface-integrated thermalization rate increases through
  the measured interval. It is not a direct measurement of numerical entropy production.
- Energy: each curve starts at its own first finite power sample. Its zero
  defines the integration origin and does not imply no earlier heating.

The final reverse accumulations are 0.00115971914 (5 events, t=0.15–0.20) and
0.00132976924 (21 events, t=0.12–0.20), in code units. They cover different
intervals. Over the common t=0.15–0.20 window, the values are 0.00115971914
and 0.00115671011; the latter is obtained by subtracting the finer run's
accumulation at t=0.15. Exact values are in `comparison.json`.

These are two simulation runs, not two postprocessing samplings of identical
states. Analysis times constrain solver steps, and the runner also sets dt_max
from t_end/num_analysis. Consequently, differences cannot be attributed solely
to trapezoidal quadrature. This comparison is not a cadence-convergence or
spatial-convergence study. No missing pre-detection energy is estimated.

All detected reverse continuity flags pass, but those flags are heuristic;
identity labels assume a single-source radial geometry. The five-event run
has only two reverse detections: enough for one velocity estimate, without a
later detection to test its velocity-based prediction. The 21-event run has
nine reverse detections, exercising that prediction from the third detection
onward. Small radius increments relative to dx=1/64 limit velocity interpretation.

Source results and repeatable local checks:

- [`../online_temporal_iwr_compgpu3_n064/README.md`](../online_temporal_iwr_compgpu3_n064/README.md)
- [`../online_temporal_iwr_compgpu3_n064_cadence21/README.md`](../online_temporal_iwr_compgpu3_n064_cadence21/README.md)

The original imported files are preserved. Local SHA256 manifests detect later
changes; they do not substitute for an independently generated remote manifest.
