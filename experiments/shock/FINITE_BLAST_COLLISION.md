# Finite blast collision: independent evidence and detection screening

## Question and result

Do finite blasts produce weakening reflected fronts, and can a missing
detection be explained by a genuine drop below Mach 1.3?

**This experiment observes weakening but does not establish a threshold
crossing. It also exposes minimum-jump screening misses beside independently
motion/jump-supported fronts above 1.3.** This is a more specific result than
assuming every late detection gap is fading or a temperature-normal failure.
It is not a physical verdict about the original 3D bubble layer.

## Finite-energy setup

`finite_blast_collision.py` evolves two symmetric hot patches centered at
x=0.4 and 0.6, each with half-width 0.02. Initially rho=1 and u=0 everywhere;
pressure is 100 inside the patches and 1 outside. Gamma is 5/3. Partial cells
use exact pressure/thermal-energy volume averages. All grids therefore start
with mass 1 and total energy 13.38 in the unit interval.

There is **no continuing energy source and no replenished post-shock reservoir**.
The same default HLL/minmod/RK2 choices and CFL 0.4 as the hydro bubble setup
advance the gas. Runs use 512/1024/2048 cells, 301 saved events to t=0.06, and
the existing 3D finder on unchanged `(N,8,8)` extrusions.

Interacting finite-pressure blasts are a standard Euler test family; see the
[Clawpack Woodward–Colella example](https://www.clawpack.org/gallery/_static/amrclaw/examples/euler_1d_wcblast/README.html).
Our symmetric gamma=5/3 hot-patch problem is a separate controlled experiment,
**not** a reproduction of that gamma=1.4, asymmetric, reflecting-wall benchmark.
There is no exact closed-form solution asserted for our evolving blast collision.

## Evidence independent of the finder

`blast_front_evidence.py` uses pressure-gradient features, their motion and
sampled primitive states. It does not read production normals, temperature
directions, zones, surfaces, or Mach estimates when forming evidence.

The observation band is fixed at 0.3<x<0.7. A central pressure rise above twice
the ambient value marks an **interaction-onset proxy**, not an exact collision
time. At each side, only a pressure peak at least twice as strong as the next
competing peak is retained; comparable peaks leave a gap. Mutual-nearest links
are bounded by local wave speeds and a cell allowance. A reflection candidate
must be born within eight cells of the midplane, persist at least 17 events,
and move outward by at least eight cells. These are local experimental feature
tracks, not replacements for the thesis geometric group tracker.

Short/long local speed fits use 9 and 17 consecutive actual event times. State
pairs at independent upstream/downstream distances of 3/4/6/8 cells must stay
on the same side of the midplane. They are checked for compression, increasing
sampled entropy, shock-frame sonic ordering, mass/momentum/enthalpy balance,
and the density-jump relation. Conservation/jump residual tolerance is 10%;
sampled entropy-signal relative error tolerance is 50%. At least three state
pairs must pass. Mass-inferred speed is not used as independent motion evidence.

Mach envelopes combine the sampled-pair spread with declared speed sensitivity:
half-cell position allowance, three fit scatters, and short/long-fit difference.
They are sensitivity estimates, **not rigorous total bounds or confidence
intervals**. An event is above threshold only if its lower envelope exceeds
1.3, and below only if its upper envelope is below 1.3. A supported crossing
requires persistent consecutive above events followed by persistent consecutive
below events on a retained track. Unverified or ended tracks are not fading.

### Controls and one rejected tracking version

| Independent-evidence control | Supported observations | Outcome |
|---|---:|---|
| Exact reflected Mach 1.914854 | 128 | Correctly above 1.3 |
| Exact reflected Mach 1.181692 | 128 | Correctly below 1.3 |
| Translated smooth isentropic compression | 0 | No shock-supported claim |
| Previously validated evolved Mach-3 collision | 28 | Correctly above 1.3 on both sides |

The smooth history is a kinematic negative control, not a claimed exact Euler
evolution. These controls calibrate the evidence rules; they do not establish
universal shock classification accuracy.

The first nearest-peak version passed analytic controls but fractured on the
evolved control because small numerical peaks stole links. It was rejected.
The dominance rule was added and all controls rerun before interpreting the
final finite-blast results. The original run is preserved outside the repository
at `/private/tmp/finite_blast_nearest_rejected_i0zsxcu_/`. Production thresholds
and all sampled-jump tolerances stayed fixed.

## Recorded outcomes

| Cells | Retained central-born outward tracks | Above-supported front-events | Without a matching detection | Below-supported events |
|---|---:|---:|---:|---:|
| 512 | 2 | 64 | 4 | 0 |
| 1024 | 2 | 98 | 12 | 0 |
| 2048 | 2 | 122 | 24 | 0 |

Counts are repeated front observations, **not numbers of distinct physical
shocks**, accuracy rates, or a claim that the denominator contains every shock.
More supported events at higher resolution do not imply worse physical accuracy
from the raw miss count alone.

Supported Mach values reach about 2.2 early in the 2048-cell run and decline
through approximately 1.6 to 1.51 in later supported intervals. An intermediate
interval is unverified and must not be connected as measured evidence. Use
`review/mach_supported_intervals.png`, which breaks curves/envelopes at gaps;
the initial overview plot is only a quick summary of retained points.

**No robust below-1.3 phase was observed on these tracks.** The tracks eventually
leave the fixed measurement band or lack support. Neither circumstance establishes
physical disappearance, fading below threshold, or absence of fronts elsewhere.

Boundary states remain undisturbed. Maximum normalized integral change is
4.27e-5, within the declared 5e-5 check; float32 drift increases with step count.
Symmetry is retained to about 4e-5. At six fixed comparison times, the 1024-grid
conserved-profile difference from the conservatively coarsened 2048 result is
smaller than the 512-grid difference. This supports refinement consistency;
the finest grid is **not exact truth**. Conserved-variable interpolation to
common times is used only for this comparison, never as detector input.

## Concrete screening failure

The read-only review verifies the stage-mask intersections on every saved cell.
All 40 above-supported misses have compression/alignment candidates nearby but
no joint minimum-jump pass and no shock zone. This result is local to these
finite-blast observations, not an explanation automatically transferred to
the older bubble alignment rejection.

The reproducibly selected example is the earliest left miss at the highest
resolution: 2048 cells, t=0.0258, x about 0.349747.

- Independently measured/sampled Mach: **1.541–1.547**; declared sensitivity
  envelope approximately **1.485–1.603**, still above 1.3.
- All 16 sampled side pairs pass the independent jump checks. Sampled pressure
  ratios are about 2.66–2.74 and entropy increases. Mass residuals are about
  0.8–1.5%, momentum residuals about 0.4–1.5%, enthalpy residuals below 0.8%.
- All four nearby cells pass compression and density–temperature alignment.
- None passes the local pressure-jump threshold; one passes the temperature
  threshold. No cell passes both, so no zone or surface is produced.

The complete sampled transition can support an above-threshold jump while the
short, fixed-separation detection stencil measures only part of it. **This
example is a minimum-jump screening failure, not reversed temperature
orientation.** Sampled jump compatibility remains conditional physical evidence,
not an exact oracle for a discontinuity or its hidden endpoint states.

## Next bounded implementation target

Use the archived finite-blast misses as regressions for the preliminary jump
measurement. Keep the final reported physical Mach threshold at 1.3. A candidate
change must recover the supported fronts while preserving the known separated
controls and recording any added contact/smooth-compression detections. Do not
lower thresholds until the desired picture appears.

That change would address this screening mechanism. It would **not by itself
fix** the earlier 3D bubble region where temperature/density alignment fails.
The original bubble must be checked separately against Tobias's two moving
slab-boundary picture.

## Reproduce

```sh
MPLCONFIGDIR=/private/tmp/master-thesis-mpl-benchmark \
python -m experiments.shock.run_finite_blast_collision \
  --resolutions 512 1024 2048 \
  --output-dir outputs/finite_blast_collision_n0512_n1024_n2048

MPLCONFIGDIR=/private/tmp/master-thesis-mpl-benchmark \
python -m experiments.shock.review_finite_blast_collision \
  --output-dir outputs/finite_blast_collision_n0512_n1024_n2048/review

python -m unittest tests.physics_modules.shock_finder.test_finite_blast_collision
```

Use fresh output directories and compatible project Python; no GPU is required.
The evolved numerical control from the preceding planar experiment is required.
Source/control-input hashes are verified before and after analysis. Actual saved
times are used for motion; requested/actual differences below 1e-6 are recorded.
Production code, Mach threshold, previous archives and bubble outputs are unchanged.
