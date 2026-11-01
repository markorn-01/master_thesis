# Reverse-candidate inspection at 128³: IWR GPU

Completed on compgpu3, physical GPU 0 (from the session), with Python 3.12.3,
JAX 0.6.2 and GPU backend. Metadata records revision
`7f49d9ed0e4f4ec9d6f907064d191c5404424f69`; the GPU model is not recorded.
The setup retains injection radius 0.0625, 21 requested events over 0–0.20,
and dt_max=0.20/21. The injection radius spans eight cells, versus four at 64³.

## Results and interpretation

| Time | Valid reverse samples | Upstream inside source | Downstream inside source | Median RH density residual |
|---|---:|---:|---:|---:|
| 0.12 | 1664 | 0 | 0 | -5.82% |
| 0.15 | 2072 | 0 | 0 | -7.55% |
| 0.20 | 2624 | 0 | 0 | -8.66% |

All inspected reverse pairs have increasing density and pressure. The pressure
increase is partly enforced by the sampler and defines the pressure-derived Mach;
it is not independent physical validation. Every selected endpoint is outside
the injection region at these times. The minimum upstream radii are approximately
0.07027, 0.07854 and 0.08927, all greater than 0.0625.

For gamma=5/3, compare directly sampled rho2/rho1 with the density ratio
predicted from each sample's pressure-derived Mach:

    J_rho,RH = 4 M² / (M² + 3)
    fractional residual = (rho2/rho1) / J_rho,RH - 1

The table reports the median per-sample fractional residual. It measures
internal jump inconsistency, not error against known exact states. At 64³ the
corresponding medians are +49.79%, +36.04% and -2.82%. Higher resolution removes
the t=0.12 source overlap and substantially reduces the early-time discrepancy.
It does not reduce the absolute median discrepancy at t=0.20. These two runs
therefore do not establish monotonic improvement, convergence, or an accuracy
bound on Mach, heating rates or accumulated energy.

Resolution changes both the numerical flow and detector sampling: fixed
cell-distance searches span different physical lengths. The same requested
times and maximum timestep do not impose identical adaptive steps across
resolutions. Source proximity, nonuniform flow and numerical broadening remain
possible influences, not effects isolated by this comparison. The physically
consistent reverse-shock interpretation should remain distinct from a claim
of quantitatively validated jump reconstruction.

See the reproducible
[resolution comparison](../reverse_profiles_iwr_resolution_comparison/README.md).

## Products, title correction and verification

`reverse_profiles.png` is the corrected 128³ profile figure. The imported
figure had a hardcoded 64³ title; it is preserved as
`reverse_profiles_original_title.png`. Only the title was corrected by
regenerating from the original CSVs, without rerunning the simulation.
The plot function now takes the resolution, and a regression test checks it.
The original run metadata and scientific CSVs are unchanged.

The figure uses shell medians for density, pressure, radial velocity and radial
flow Mach; the latter differs from pressure-jump shock Mach. Dashed lines
mark median surface radii. Shading marks injection. Colored thin lines and
markers identify one actual valid reverse-candidate sample, chosen nearest
the median surface radius. Individual samples need not lie on shell medians.

Run the NumPy-only readback verifier from the repository root:

```bash
python outputs/reverse_profiles_iwr_compgpu3_n128/verify_profiles.py
```

`local_checks.json` confirms sample counts, compression counts, endpoint flags,
timing and the defining pressure-Mach relation. All recorded source hashes
match files at the recorded Git revision. The current inspection script hash
differs because of the subsequent title fix, which is explicitly reported.
The verifier requires the recorded revision in local Git history. Full gas
states were not saved, so their values cannot be independently rechecked.
No dedicated online/offline equivalence run at 128³ is claimed here.

To regenerate the corrected figure, use the repository's compatible JAX and
Matplotlib environment (the plotting module imports the inspection runner):

```bash
PYTHONPATH=. python outputs/reverse_profiles_iwr_compgpu3_n128/replot.py
```

Three focused geometry/plot-label tests passed locally. The original and
corrected figures were visually inspected. `SHA256SUMS` records local
preservation hashes; it is not an independently generated remote manifest.
No estimator, filtering or solver changes were made.

Original simulation command (use a new directory for repetitions):

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m experiments.wind_bubble.check_reverse_profiles \
  --num-cells 128 --require-gpu \
  --output-dir outputs/reverse_profiles_iwr_compgpu3_n128_repeat
```
