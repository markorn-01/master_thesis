# GPU confirmation of reverse-candidate sampling limitations

Both normal-path inspections completed on compgpu9, physical GPU 3, at 64³ and
128³. Python 3.12.3, JAX 0.6.2, GPU backend; revision edc0b93. Source hashes,
requested/actual times and independent readback passed for both imports.
The hardware model is not recorded. Host and physical slot come from the session.

Both cases use injection radius 0.0625, 21 requested events and dt_max=0.20/21,
and inspect t=0.20. No physical estimator, selection or filtering rule changed.
The original CPU demonstrations are preserved separately; this does not claim
exact CPU/GPU equality. This is normal-path physical interpretation, not a new
128³ online/offline equivalence test.

## Confirmed findings

All selected endpoints lie outside the Boolean shock zone and injection at
this time. Independent interpolation reproduces selected pressure/density
values exactly, and original pressure-candidate choices are reproduced. No
coordinate, sign or pressure-choice mismatch was found on these inspected paths.
This is a limited implementation check, not proof that no error can exist.

The field range at s−0.5, s, s+0.5 cells, divided by the central value, is:

| Median local range | 64³ | 128³ |
|---|---:|---:|
| Upstream pressure | 64.6% | 26.8% |
| Downstream pressure | 2.68% | 2.95% |
| Upstream density | 33.0% | 13.2% |
| Downstream density | 18.6% | 2.47% |

These are variations, not physical error estimates. One cell has different
physical width at the two resolutions. Upstream neighborhoods overlap the
Boolean zone for 272/608 and 1592/2624 samples, respectively; 24 of the 64³
neighborhoods also overlap injection. No downstream neighborhood overlaps
either mask. Thus some variation includes nearby transition structure, while
downstream density can vary even with comparatively flat pressure outside
both masks. Mask exit alone is not a plateau-recovery guarantee.

All downstream samples are the farthest pressure candidate, at 1.5 cells beyond
the zone exit. Holding the original upstream sample fixed and hypothetically
using each existing downstream candidate gives median density-RH residuals:

| Hypothetical downstream offset | 64³ | 128³ |
|---|---:|---:|
| 0.25 cells | -16.3% | -11.0% |
| 1.50 cells (original choice) | -2.82% | -8.66% |

The residual uses J_rho/J_rho,RH−1, where J_rho,RH=4 M²/(M²+3), gamma=5/3,
and M is pressure-derived. No candidate is chosen to minimize that residual;
no saved physical measurement is replaced. Candidate sensitivity demonstrates
that smaller residual at 64³ is not sufficient evidence of better reconstruction.
It does not isolate the full discrepancy or prove error cancellation.

The evidence supports finite-resolution sampling limitations in varying flow,
not an estimator redesign merely to enforce density-RH agreement. Keep the
measurements and limitations explicit. Accurate early heating, completeness,
and convergence remain separate physical questions.

## Reproduce saved-data analysis

Using NumPy and Matplotlib from the repository root:

```bash
python outputs/reverse_path_diagnostics_iwr_compgpu9/derive_checks.py
python outputs/reverse_path_diagnostics_iwr_compgpu9/plot_diagnostics.py
```

The combined figure is endpoint_variation_and_sensitivity.png. It was visually
inspected, as were both original path figures. Bars show medians; whiskers and
shading show 16th–84th sample percentiles, not confidence intervals. The two
input folders preserve metadata, all path arrays, candidates, original samples,
summary CSV/JSON products and independent readback checks. Full gas states are
not retained. SHA256SUMS files record local preservation, not independently
supplied remote checksums.

The separate 64³, 21-event online/offline run also passed, including all new
quality flags and exactly unchanged legacy products; see
[../sampling_quality_iwr_compgpu9_n064/README.md](../sampling_quality_iwr_compgpu9_n064/README.md).
This completes the intended GPU confirmation before evolution visualization.
No additional runtime or memory performance claim is made.
