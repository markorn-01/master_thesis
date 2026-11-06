# Budget-matched two-bubble resolution check

## Purpose

Inspect the contact region and small inner patches at 128³ against the existing
64³ sequence, before treating graph components as physical shock identities.
This is a controlled morphology/resolution experiment, not an established
convergence sequence or a validation of temporal tracking.

## Why an explicit budget match is needed

The existing 3D EI prescription divides by the analytic sphere volume but
injects into cell centers inside R−dx/2. The discrete source volume therefore
changes with resolution, even at fixed nominal radius R=.0625.

| Per source | Existing 64³ | Unchanged-parameter 128³ | Budget-matched 128³ |
| --- | ---: | ---: | ---: |
| Active injection cells | 176 | 1780 | 1780 |
| Mass-rate parameter | .01 | .01 | .00791011236 |
| Velocity parameter | 10 | 10 | 10 |
| Actual discrete mass rate | .00656514140 | .00829968160 | .00656514140 |
| Prescribed discrete mechanical-energy rate | .32825707013 | .41498408013 | .32825707013 |

Without matching, the higher-resolution prescribed mechanical-power coefficient is 26.42% larger.
The new optional `--match-source-budget REFERENCE_METADATA` scales the shared
mass-rate parameter by .791011235955 at 128³. Energy and mass rates scale
together while the velocity parameter remains fixed. Both sources are matched.
The original default parameters and source implementation are unchanged.
Nominal luminosity in the new metadata reflects the adjusted parameter, while
the recorded discrete rates reflect the actual mask volume.

This option matches the mass injection rate and prescribed mechanical-energy
source coefficient, not their spatial distribution or cumulative net gas-energy
deposition. The existing EI conversion from energy rate to pressure rate depends
on the current density and velocity. A source-only 32³ test at rest measures a
net gas-energy rate of .65651413 for the matched total coefficient .65651414;
with uniform velocity x=1, it measures .66270645. This is recorded in
`source_energy_semantics.json`. The physical source update is unchanged. The
mask, resolved injection profile, hydro truncation errors, and sampled slice
planes still differ. Matching budgets does not prove physical convergence.
The reference metadata is validated against its recorded mask counts, source
positions/radius, velocity and source prescription. Inconsistent rates are
rejected. The runner also checks that its constructed geometry agrees with
the preflight mask counts.

## Fixed setup and analysis schedule

Keep sources at (.35,.5,.5) and (.65,.5,.5), injection radius .0625, unit cube,
open boundaries, rho=1, P=.01 and gamma=5/3. Use 17 events over t=0–.16, retaining
the reference event schedule and dt_max convention. The new reference-matching
option itself controls source budgets only, not arbitrary other settings.

Inspect .04, .08, .12, .13, .14 and .16. Check actual analysis times, source
overlap flags, finite measurements and boundary clearance before interpretation.
No full gas-state sequence or offline reference run is requested at 128³;
the stored-state verification option remains restricted to at most 64³.

## Local validation before GPU execution

- Five NumPy tests check counts/rates, default parameters, matching both sources,
  rejection of incompatible references and solver/preflight count disagreement.
- Ten JAX tests check original injection behavior, two-source products,
  actual mass/energy increments at rest under matched parameters, and the
  state-dependent net-energy limitation.
- A 32³ three-event run to .01, using the new matching option, passes online–
  offline comparison for all 21 history fields, exact histograms, catalogs,
  slices and final gas state. This is workflow validation, not the planned
  physical comparison.

The preflight result is in `outputs/two_bubbles_resolution_preflight_n064_n128/`.
It verifies reference settings/times and the actual injection/configuration
module hashes. The original runner hash is historical and legitimately changes
with the new explicit CLI option. The 128³ GPU run is still pending.

## Remote run: compgpu3, physical GPU 0

After committing/pushing the preparation, connect and select the user-confirmed
GPU. Do not change another user's processes if availability has changed.

```bash
ssh mngo@compgpu3.iwr.uni-heidelberg.de
```

```bash
cd ~/master_thesis && git pull --ff-only && source .venv-iwr/bin/activate && \
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false && \
python -m experiments.wind_bubble.check_gpu_environment
```

Run the focused checks and then the simulation:

```bash
python -m unittest discover -s tests/experiments -p test_two_bubble_source_budget.py -v && \
python -m unittest discover -s tests/experiments -p test_two_bubbles.py -v && \
python -m experiments.wind_bubble.run_two_bubbles \
  --num-cells 128 --num-analysis 17 --t-end 0.16 \
  --match-source-budget outputs/two_bubbles_iwr_compgpu9_n064/metadata.json \
  --require-gpu \
  --output-dir outputs/two_bubbles_budget_matched_iwr_compgpu3_n128
```

This output directory must be new. There should be one sequence of 17 analysis
messages because offline verification is not enabled. The metadata should
finish with `status=complete`, 17 actual times, counts [1780,1780] and per-source
discrete energy rates near .32825707013. The GPU may appear as JAX device 0
after CUDA visibility remapping; physical GPU selection is controlled by the
environment variable.

## Subsequent comparison

Use common pressure/density and Mach color scales, and label the actual slice
planes: 64³ saves z=.5078125; 128³ saves z=.50390625. These are nearby cell-center
planes, not exactly the same interpolated midplane. Do not compute a pointwise
field error or assume sample-to-sample correspondence from those slices.

Report shock sample distributions, source overlap, clearance, component sizes
and patch geometry at matching times. For graph sensitivity, distinguish a
radius fixed in cells from a radius fixed in physical code units. A 20-sample
cutoff also does not represent the same physical patch size at both resolutions;
show raw components and cutoff sensitivity instead of claiming component counts
alone establish physical shock counts or convergence.

If the smaller patches persist at higher resolution, investigate the spatial
representation and clustering. If they change, report that dependence rather
than declaring the high-resolution grouping automatically correct. FuzzyCat
membership corrections and the unresolved temporal representation remain
separate questions.
