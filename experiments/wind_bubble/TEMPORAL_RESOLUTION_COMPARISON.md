# Temporal support at 64³ and prescribed-budget-matched 128³

## Purpose and scope

Test whether the more connected inner surfaces at 128³ also produce more
consistent adjacent-event candidate correspondences. This reuses saved catalogs
and checked graph partitions; no new hydrodynamic simulation, physical shock
identifier, or production tracking change is introduced.

Inputs are the two runs documented in `TWO_BUBBLE_RESOLUTION_RESULTS.md` and
`outputs/two_bubbles_resolution_comparison_n064_n128`. The prescribed mass and
mechanical-energy coefficients match, not necessarily net deposited gas energy.

## Method

All 16 adjacent intervals in the 17-event sequence are evaluated at both
resolutions, using three spatial grouping settings and 30°/60° normal limits.
The settings are inherited from the spatial comparison:

- Native radius 1.5 cells, cutoff 20 at both resolutions.
- Native radius 1.5 cells, cutoff 20 at 64³ and 80 at 128³.
- Common radius 1.5/64 code units, with cutoffs 20/80.

Scaling the count cutoff by N² is a sampling heuristic, not an area measurement.
Previously excluded components remain in the saved raw partitions.

Association gates are **common physical distances** 1.5/64 and 2.5/64 code
units: 1.5/2.5 native cells at 64³ and 3/5 native cells at 128³. These are geometric
search distances, not measured propagation speeds. Signed-normal compatibility
is limited to 60°. Each parent needs at least 50% covered samples, and its best
child must exceed the runner-up by at least 0.25 in parent coverage. The unique
union of eligible parents must cover at least 50% of the child. This is the
existing asymmetric pooled-support experiment, not a general split/merger rule.

The full audit has 384 comparisons. Configurations can yield identical partitions
(e.g. all three spatial settings at 64³); these are sensitivity configurations,
not independent trials. Parent choices, runner-up scores, child unions, accepted
links and exclusive/redundant support are saved. No source positions or Mach
numbers enter matching. Source-overlap flags remain in the input catalogs and
are not used to reject links.

## Findings

For the illustrated 60° spatial limit:

- At 64³, the 20-sample cutoff hides the inner patches at t=.12. At .13, ten
  inner patches become retained but have no accepted parent at .12. Their
  appearance is not evidence of physical shock birth. Pooling subsequently
  links the fragments into two larger inner groups at .14.
- At 128³ with cutoff 80, three large retained groups have three separate
  accepted links on every interval from .11→.12 through .15→.16, under both
  distance gates and both spatial angle limits. The two large inner groups
  first enter this retained sequence at .11 without accepted parents from .10.
  This is improved geometric continuity of the retained representation; it does
  not establish physical ancestry or guarantee that early detections are complete.
- The native 128³ graph with cutoff 20 still has residual fragments. Two new
  32-sample components at .13 have no accepted parents from .12. At .13→.14,
  the narrower gate accepts five links, while the wider gate makes two parent
  choices ambiguous and leaves two child groups unsupported.
- At .14→.15, both gates accept five links into three children in that native
  cutoff-20 configuration. Under the wider gate, two accepted links add **zero
  exclusive child samples**: their support is already supplied by other parents.
  Thus a wider search can increase ambiguity and redundancy, even at higher
  resolution. It should not be chosen solely to maximize accepted links.
- Stable group counts with cutoff 80 do not erase the small components. At .14,
  the native graph excludes 104 samples, whereas the common-physical-radius
  graph connects them into retained groups. Both show three groups.

The retained large groups are promising candidates for a reviewed evolution
view, but the result does not validate a general shock tracker. The existing
near-parallel-sheet control still accepts a stipulated unrelated parent. Its
known false link is retained explicitly; it is not a measured simulation error
rate. The simulation has no independent physical identity ground truth here.

## Verification and figure

`outputs/two_bubbles_temporal_comparison_n064_n128/` contains:

- `summary.csv`: group, link, unsupported/ambiguous/redundant counts and sample
  accounting for every setting and interval.
- `support_records.json.gz`: complete parent/child decisions, unique child-row
  support lists, and candidate links. Group numbers are local to each event.
- `temporal_support_comparison.png`: retained groups, groups without accepted
  support, and redundant accepted links versus child time. Columns distinguish
  the two physical distance gates. All 30° results are in the tables.
- `controlled_examples.json`, `metadata.json`, `readback.json`, `SHA256SUMS`.

All 384 saved summaries pass readback. An independent per-point neighborhood
query checks 36 selected comparisons (.12→.13, .13→.14, .15→.16, both resolutions,
all three spatial settings, both gates at the 60° spatial limit). It reproduces
148 parent decisions and 164 child unions, including exclusive-support counts.
Nineteen existing pooled-support, evidence and association tests pass. Input
hashes are unchanged. These checks validate the calculations, not physical IDs.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.wind_bubble.compare_resolution_associations \
  outputs/two_bubbles_resolution_comparison_n064_n128 \
  outputs/two_bubbles_iwr_compgpu9_n064 \
  outputs/two_bubbles_budget_matched_iwr_compgpu3_n128 \
  --output-dir outputs/two_bubbles_temporal_comparison_n064_n128_new

/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.wind_bubble.verify_resolution_associations \
  outputs/two_bubbles_temporal_comparison_n064_n128_new
```

The output directory must be new. NumPy, SciPy and Matplotlib suffice; neither
JAX nor FuzzyCat is needed. Review the accepted spatial links and unsupported
fragments before assigning any physical interpretation. A smooth surface mesh
and the strong density-ratio resolution dependence remain separate open tasks.
