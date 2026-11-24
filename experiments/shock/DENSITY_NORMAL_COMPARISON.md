# Bounded density normal detector comparison

**Decision: this alternative fails the agreed acceptance gates and should not
be applied to the saved bubble simulation or promoted to production.**
The comparison isolates a state-separation problem. Removing the temperature
gradient from candidate selection finds the six known missed fronts, but the
fixed side averages include contact gas instead of the immediate shocked gas.
Accepting those kinematic candidates alone also accepts smooth compression.

## The single alternative tested

The planar hydro analogue draws on the candidate geometry and averaging in
[SHOCKFIND, Section 3.1](https://arxiv.org/pdf/1608.02050):

1. Search compression/density-gradient candidates and local compression maxima.
2. Use the density gradient for orientation, without a temperature-gradient test.
3. Average three samples on each side, at distances of 2, 3, and 4 cell widths.
4. Infer a shock-frame speed from density and velocity averages.
5. Require increasing pressure and density, converging normal velocity,
   upstream Mach at least 1.3, and upstream-supersonic/downstream-subsonic flow.
6. Accept a hydro jump only when the sampled entropy change is positive.

The magnetic fast/slow classification from SHOCKFIND is not used; these are
hydrodynamic inputs with gamma=5/3. This is our declared hydro analogue,
not a run or validation of the published SHOCKFIND MHD implementation.
Transverse cylinder averaging is trivial for the planar inputs. The mass-based
speed is inferred from the same states, not independently tracked motion.
Sound speed uses pressure and density; their temperature gradient is unused.

All settings are fixed across cases. There is no per-case distance choice and
no strict 5%, 10%, or 20% Rankine–Hugoniot residual veto. The comparison keeps
kinematic candidate counts separate from accepted-jump counts.

## Inputs and acceptance gates

The comparison uses the frozen 228-case production baseline: 180 shock/contact
pairs, 12 isolated shocks, and 36 moving contacts, on 32- and 64-cell planar
profiles. Exact conservative grid averages and synthetic 3-/5-cell broadening
retain their original labels. Of the six original zone rejections, two use
exact conservative averages and four use synthetic broadening.

Six additional controls cover four smooth isentropic compressions, a stationary
pressure bump, and uniform-pressure compression. The unchanged production
finder is also evaluated on these additional inputs. The smooth profiles
are compressive initial states with no entropy jump at the sampled instant;
their subsequent evolution is not asserted to remain shock-free.

The alternative may advance only if it recovers all six known zone-rejected
shocks, rejects all contact/smooth controls, preserves exact isolated-shock
locations and Mach numbers within 5%, and loses no previously recovered pair
detection. The full case matrix determines the result; illustrative figures
do not select the passing cases.

## Results

| Check | Result |
| --- | --- |
| Six previously zone-rejected shocks | All six located kinematically; none accepted after entropy check |
| 12 isolated shocks | All detected; exact isolated Mach controls preserved |
| 36 pure contacts | No accepted detections |
| Six smooth/background controls | No accepted detections; four kinematic false candidates |
| Previously recovered pair cases | 36 lose an accepted detection under this fixed alternative |
| Overall acceptance | Failed |

The alternative accepts correctly located fronts in 58 of 180 pair cases.
This is not directly comparable with the production baseline's 74 fully
recovered pairs, whose outcome additionally includes Mach accuracy. The
36 lost detections are computed specifically within those previously fully
recovered cases. Baseline surface-sample totals are 3D counts; alternative
totals are 1D sample counts. Decisions compare case-level presence and
location, not those raw sample totals.

Two steep smooth controls also generate surfaces in the unchanged production
finder. That is a limitation exposed by these constructed inputs, not a
measured false-positive rate in the wind-bubble simulations.

## Why the known physical shock is still rejected

For the dense-contact half-cell example, the alternative uses:

| State | Density | Velocity | Pressure |
| --- | --- | --- | --- |
| Sampled upstream | 1 | 0 | 1 |
| True immediate downstream, independently known | 3 | 2.581989 | 11 |
| Sampled downstream contact gas | 24 | 2.581989 | 11 |

The density normal now locates the pressure transition correctly. However,
the side average is contact gas. Its sampled entropy change is
`ln(11) - (5/3) ln(24) = -2.898861`, so the hydro entropy gate rejects it.
The actual shock heats the immediate downstream gas and has positive entropy
change. Entropy admissibility is a physical requirement of Euler shocks;
see the [Clawpack Euler chapter](https://www.clawpack.org/riemann_book/html/Euler.html).
The incorrectly paired samples are not the two states of that shock.

The mass-inferred Mach is also biased to 2.086957 instead of 3, because it uses
the contact density. Thus finding a candidate does not establish a correct
shock-state reconstruction. Dropping the entropy gate admits the six target
candidates but also four smooth-compression controls. It fails the agreed
false-detection requirement.

## What this comparison establishes

This fixed density-normal/averaged-state alternative is not a validated fix.
It rules out this particular replacement under these settings; it does not
rule out every density-based method or establish that the bubble's missing
features are physical shocks. The remaining algorithmic issue is recovering
separate shock and contact states when a side average mixes them.

No bubble replay was performed after the gate failed. No production finder,
Mach estimator, callback, grouping, visualization, or thermalization output
was changed. Historical results remain intact.

## Reproduction and validation

```bash
python -m experiments.shock.check_density_normal_detector \
  --output-dir outputs/density_normal_detector_comparison_new

python -m unittest discover -s tests/physics_modules/shock_finder -p 'test_*.py'
```

Use a fresh directory. The result files are in
`outputs/density_normal_detector_comparison_n032_n064/`:
`verification.json`, the complete `summary.csv`, per-candidate `candidates.csv`,
and `detector_comparison.png`. Direct input files and the frozen production
sources are fingerprinted. The comparison ran locally on CPU with JAX 0.6.2.
All 89 shock-finder tests passed, including ten new detector and acceptance-gate
tests. Passing implementation tests does not override the failed experimental
acceptance result.
