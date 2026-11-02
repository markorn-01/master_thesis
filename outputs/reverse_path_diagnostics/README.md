# Investigation of reverse-candidate sampling at t=0.20

Question: why is the pressure-Mach/density-jump residual closer to zero at 64³
than at 128³ at the late inspected time? The original GPU inspections saved
endpoints, not full gas states or normal paths. This investigation therefore
uses **new CPU reruns** at both resolutions, with the same physical setup,
21-event schedule, injection radius 0.0625 and dt_max=0.20/21. It is not a
reconstruction of GPU gas states or an exact CPU/GPU equivalence test.

No detector, Mach estimator, source prescription, candidate selection or energy
rule changed. All results concern the existing profile-aware estimator.

## Findings

1. **No coordinate/sign/pressure-choice mismatch was found in the inspected
   paths.** Independent interpolation reproduces all selected pressure/density
   values; maximum absolute differences are 2.98e-8 at 64³ and 8.94e-8 at 128³.
   Reconstructed zone exits and all six pressure candidates reproduce the
   analyzer's upstream and downstream distances.
2. **Outside the Boolean shock zone does not imply a flat sampled state.** All
   selected endpoints lie outside both the Boolean zone and injection at t=0.20.
   Pressure/density still vary around those points. A neighborhood around a
   point outside a mask can overlap the masked region.
3. **Density consistency is sensitive to where downstream density is sampled.**
   Every inspected downstream pressure choice is the farthest existing candidate
   at 1.5 cells beyond the zone exit. Changing that hypothetical choice while
   retaining the original selected upstream state changes the median residual
   substantially at 64³. These alternatives are not applied to measurements.

### Local variation

The robust local variation is (maximum−minimum)/|central value| using field
values at signed positions s−0.5, s, s+0.5 cells around the selected endpoint.
This includes curvature around a minimum, unlike a signed first derivative or
just the difference between the outer samples. Table entries are population
medians, not errors against exact physical states or calibrated quality thresholds.

| Median local range | 64³ | 128³ |
|---|---:|---:|
| Upstream pressure | 64.6% | 26.7% |
| Downstream pressure | 2.68% | 2.95% |
| Upstream density | 33.0% | 13.2% |
| Downstream density | 18.6% | 2.47% |

A one-cell interval has different physical width at the two resolutions; this
is a local resolution-scale diagnostic, not a comparison at fixed physical
separation. Of the upstream neighborhoods, 272/608 at 64³ and 1592/2624 at
128³ include a Boolean-zone sample; 24/608 at 64³ also include an injection
sample. None of the downstream neighborhoods overlap either mask. Therefore
these variations cannot all be labeled pure background-gradient measurements.
The downstream density variation remains appreciable at 64³ even without
mask overlap and with fairly flat downstream pressure.

### Candidate sensitivity

For each original selected upstream state, the six original downstream pressure
candidates were hypothetically used to recompute pressure-derived Mach and the
density-RH residual. No alternative was chosen to minimize the residual.

| Hypothetical offset beyond downstream zone exit | 64³ median residual | 128³ median residual |
|---|---:|---:|
| 0.25 cells | -16.3% | -11.0% |
| 1.50 cells (original selected choice) | -2.83% | -8.66% |

All six offsets and population percentiles are in `post_candidate_sensitivity.csv`.
The RH density prediction is 4 M²/(M²+3), gamma=5/3; pressure defines M, while
density is sampled independently at those pressure-selected positions.

## Interpretation and decision

The checked implementation behaves as designed. The evidence supports a
finite-resolution sampling limitation in nonuniform flow, including residual
transition structure and density gradients near pressure-selected states.
In particular, closer density-RH agreement at 64³ is sensitive to sampling
position and is not sufficient evidence of better physical accuracy. Error
cancellation is a possible interpretation, not a proven decomposition of the
remaining discrepancy. This inspection does not independently quantify true
Mach error, isolate the cause of the full residual, or establish convergence.

**No estimator redesign or automatic filtering is justified by this check
alone.** Keep the measured values and their limitations explicit. Do not tune
the pressure sampler just to force density-RH agreement. The next confirmation
is the same path inspection on the GPU; only after that should changes to
sampling or quantitative physical-validation claims be considered.

## Figures and reproducibility

`endpoint_variation_and_sensitivity.png` summarizes population variations and
hypothetical candidate sensitivity. Bars show medians; whiskers and shading
show 16th–84th sample percentiles, not confidence intervals. Individual
median-residual illustrative paths are in:

- [`../reverse_paths_cpu_n064/normal_path.png`](../reverse_paths_cpu_n064/normal_path.png)
- [`../reverse_paths_cpu_n128/normal_path.png`](../reverse_paths_cpu_n128/normal_path.png)

The plots and readback were inspected locally. Each case preserves all paths
in `normal_paths.npz`, per-sample `endpoint_checks.csv`, a summary, run metadata
and readback checks. No full gas-state series is saved. Six focused geometry,
interpolation and profile-plot tests passed. Source hashes match inspected
files; the metadata commit is the baseline revision with working-tree changes,
not a claim that the new inspection script was already committed.

Recheck and regenerate with NumPy and Matplotlib from the repository root:

```bash
python outputs/reverse_path_diagnostics/verify_paths.py outputs/reverse_paths_cpu_n064
python outputs/reverse_path_diagnostics/verify_paths.py outputs/reverse_paths_cpu_n128
python outputs/reverse_path_diagnostics/derive_checks.py
python outputs/reverse_path_diagnostics/plot_diagnostics.py
```

For a new simulation or GPU command, see
[`REVERSE_SAMPLING_PATHS.md`](../../experiments/wind_bubble/REVERSE_SAMPLING_PATHS.md).
`SHA256SUMS` files are local preservation manifests.
