# Reverse-candidate radial inspection: IWR GPU

Completed on compgpu3, physical GPU 0 (selection recorded in the session),
using `.venv-iwr`, Python 3.12.3 and JAX 0.6.2. Metadata records the GPU
backend and revision `7727e7d2ac1944b0be8ebb2e0b04bb98646ab3e6`; the GPU
model is not recorded. Results were copied and checked locally on 2026-11-01.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m experiments.wind_bubble.check_reverse_profiles \
  --num-cells 64 --require-gpu \
  --output-dir outputs/reverse_profiles_iwr_compgpu3_n064_repeat
```

This is a new simulation with the same 21 requested times, dt_max=0.20/21,
physical injection radius 0.0625 and baseline wind setup as the temporal run.
Full gas states are retained temporarily for inspection, not saved to disk.
No solver, source prescription, shock physics or estimator was changed.

## Figure

`reverse_profiles.png` shows density, pressure, radial velocity and radial
flow Mach as spherical shell medians at t=0.12, 0.15 and 0.20. Dashed lines
mark median reverse/forward candidate radii. Gray shading marks injection.
Thin colored lines show a single valid reverse sample's upstream/downstream
radii; markers show its actual interpolated density and pressure. The point
nearest the reverse population's median radius is selected deterministically.
Individual samples need not lie on the shell median curves.

The final row also shows reverse-candidate pressure-jump Mach values. They
are distinct from radial flow Mach in the simulation frame. Pressure-derived
shock Mach is not an independent measurement of shock propagation speed.

## Findings

| Time | Valid reverse samples | Density-increasing pairs | Upstream inside injection | Downstream inside injection |
|---|---:|---:|---:|---:|
| 0.12 | 48 | 48 | 48 | 0 |
| 0.15 | 336 | 336 | 0 | 0 |
| 0.20 | 608 | 608 | 0 | 0 |

Every pair has increasing pressure too. This is partly enforced by pressure
selection and used to infer Mach, so it is not independent shock validation.

At t=0.12 all upstream endpoints lie at r approximately 0.059493, inside the
prescribed source radius 0.0625, although their detected surface lies outside.
Excluding source-region surface cells therefore does not exclude source-region
state samples. At t=0.15 and 0.20 both endpoints lie outside injection.

Compare the directly sampled density ratio with the ideal-gas Rankine–Hugoniot
prediction for gamma=5/3, using each sample's pressure-derived Mach:

    J_rho,RH = 4 M² / (M² + 3)

| Time | Median sampled density ratio | Median predicted ratio | Median fractional residual |
|---|---:|---:|---:|
| 0.12 | 2.32659 | 1.55323 | +49.8% |
| 0.15 | 2.57806 | 1.83901 | +36.0% |
| 0.20 | 2.00242 | 2.07657 | -2.8% |

The residual is the median of per-sample ratios J_rho/J_rho,RH - 1, not the
ratio of the two reported medians. It is a diagnostic discrepancy, not an
error measured against known exact physical states. Source proximity,
numerical broadening and nonuniform flow may affect these comparisons; this
inspection does not isolate their contributions.

The profiles, inward normals and compression support a reverse-shock
interpretation. They do not establish accurate early jump reconstruction.
Leaving the injection region is necessary context, but the substantial
residual at t=0.15 shows it is not sufficient evidence of accurate sampling.
The small median residual at t=0.20 is encouraging without establishing
convergence or agreement for every sample.

Early reverse-candidate thermalization rates and accumulated energies inherit
these sampling limitations. No samples were removed, integration origins
changed, or measured values corrected in response to this inspection.

## Verification and preserved products

Run with NumPy from the repository root:

```bash
python outputs/reverse_profiles_iwr_compgpu3_n064/verify_profiles.py
```

`local_checks.json` records passing checks: all three recorded source hashes
match; requested times equal actual times; saved sample counts, compression
counts and injection flags are reproduced from CSV; pressure-derived Mach
satisfies its defining relation. The latter is a consistency check, not an
independent validation. The saved figure was visually inspected.

A separate comparison with the earlier 21-event GPU catalogs is recorded in
`temporal_catalog_comparison.json`. At the three inspected events all surface
indices and validity flags match, and all sampled densities and Mach values
match exactly. Radius differences are below 4.2e-8 code units and pass
rtol=1e-6, atol=1e-8; the inspection computes geometry on device whereas the
online catalog computes it on the host. Full gas states were not compared.

`profile_*.csv` stores shell medians; `samples_*.csv` stores all detected
outside-source surface samples; `sample_checks.csv` stores count summaries.
Metadata and original imported products are preserved. `SHA256SUMS` is a
local preservation manifest, not an independently supplied remote checksum.
The earlier CPU demonstration remains separate evidence, rather than a claim
of exact CPU/GPU equivalence.

Next useful physical check: repeat this inspection at 128³ while keeping
injection radius and event schedule fixed. Compare source proximity and jump
consistency at the same times before interpreting early reverse-shock energy
as quantitatively validated. Two resolutions alone do not establish convergence.
