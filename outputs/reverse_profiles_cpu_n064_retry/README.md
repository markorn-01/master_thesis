# Reverse-candidate thermodynamic inspection: CPU demonstration

Completed locally on 2026-11-01 with Python 3.12.4, JAX 0.6.2, CPU backend.
This is a NEW simulation using the same setup, 21-event schedule and dt_max
as the GPU temporal run. It is not a reconstruction of saved GPU gas states,
and no CPU/GPU equivalence claim is made. The full states were not in the
imported catalogs, so a rerun was necessary.

## Figure and findings

Open `reverse_profiles.png`. Columns are t=0.12, 0.15 and 0.20. Rows show
spherical median density, pressure, radial velocity and radial flow Mach.
The last row also plots profile-aware reverse-candidate shock Mach samples.
The two Mach definitions differ: flow Mach uses simulation-frame velocity,
whereas shock Mach is inferred from pressure contrast.

The shaded region has r <= 0.0625. Dashed lines mark median detected radii.
Thin vertical lines locate one illustrative valid upstream/downstream pair;
markers show its actual interpolated density and pressure. The sample nearest
the reverse population's median radius is selected deterministically. Its
values need not lie on the spherical shell median curves.

| Time | Reverse samples | Density-increasing pairs | Upstream inside injection | Downstream inside injection |
|---|---:|---:|---:|---:|
| 0.12 | 48 | 48 | 48 | 0 |
| 0.15 | 336 | 336 | 0 | 0 |
| 0.20 | 608 | 608 | 0 | 0 |

All these pairs have valid numerical sampling and increasing pressure. This
pressure condition is partly enforced by the sampler and used in Mach
reconstruction, so it is not an independent validation of shock Mach.

**At t=0.12, every upstream endpoint is inside the prescribed injection
region**, at radius approximately 0.059493. The detected surface itself is
outside, at radius approximately 0.082074. Excluding surface cells within the
source therefore does not exclude source-region upstream samples. This limits
the earliest measurement's interpretation. At later inspected times both
endpoints are outside the source, but that alone does not prove plateau recovery.

The surrounding profiles show a radial velocity drop and pressure/density
increases around the inner candidate, supporting a reverse-shock interpretation.
The independently sampled density contrast is also compared with the ideal-gas
Rankine–Hugoniot density ratio predicted from pressure-derived Mach:

    J_rho,RH = 4 M² / (M² + 3)    (gamma = 5/3)

Median per-sample fractional differences, J_rho/J_rho,RH - 1, are approximately
+49.8%, +36.0%, and -2.8% at the three times. The later agreement improves in
this demonstration, while the early measurements are not consistent with
an accurately recovered idealized jump. Finite-resolution broadening,
nonuniform upstream/downstream flow and source proximity remain relevant;
this test does not isolate their effects or establish convergence. Pressure/Mach
agreement is by construction and is checked only as a readback sanity check.

## Reproduction and verification

```bash
MPLCONFIGDIR=/tmp/thesis-cache/matplotlib \
/tmp/thesis-sparse-venv/bin/python -m experiments.wind_bubble.check_reverse_profiles \
  --num-cells 64 --output-dir outputs/reverse_profiles_cpu_n064_repeat
```

The temporary interpreter path is local to this session. The same module works
in `.venv-iwr`; add `--require-gpu` for the remote check. See
[`REVERSE_PROFILES.md`](../../experiments/wind_bubble/REVERSE_PROFILES.md).

`profile_*.csv` contains shell medians; `samples_*.csv` contains all outside-source
surface samples, including their actual selected thermodynamic states and
endpoint radii. `sample_checks.csv` summarizes counts. No full gas states are
written. Recheck CSV counts, source hashes, requested/actual times, and derived
jump comparisons using NumPy:

```bash
python outputs/reverse_profiles_cpu_n064_retry/verify_profiles.py
```

`local_checks.json` records passing readback checks. These checks validate the
saved products' consistency; they do not assert that every physical diagnostic
is accurate. Both focused geometry tests passed:

```bash
python -m unittest tests.experiments.test_reverse_profiles
```

They check upstream/downstream signs, inclusion of refined offsets, source
endpoint flags, and empty detections. The initial attempt in the neighboring
`reverse_profiles_cpu_n064` directory failed before integration because the
new inspection runner passed an extra positional argument to the integrator.
That call was corrected before this completed run; the original failure metadata
is retained. No solver or shock-finder changes were made.

`SHA256SUMS` contains local preservation hashes, excluding itself.
