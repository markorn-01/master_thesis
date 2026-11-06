# Candidate temporal surface associations

## Why not feed catalog row IDs directly to FuzzyCat?

FuzzyCat compares cluster memberships over different representations of the
same object set. Its files contain object indices or membership probabilities;
cluster similarity is based on shared membership (Jaccard similarity).
See the [official API](https://fuzzycat.readthedocs.io/en/latest/api.html) and
[algorithm description](https://fuzzycat.readthedocs.io/en/latest/howitworks.html).

Our surface catalogs are new detections at each event: their lengths differ,
and row 100 at one time need not describe the same surface location, material,
or physical structure as row 100 at the next time. Grid-cell IDs would describe
fixed spatial locations, not persistent shock samples as the front moves.
Consequently a direct row-ID input would produce artificial membership overlap.
This does not rule out FuzzyCat; it identifies a representation problem to solve
before applying it across time. FuzzyCat can also be tested on parameter/order
ensembles at one event, where original catalog row IDs already provide a common
object set. That would measure robustness to clustering choices, not temporal
tracking. FuzzyCat has not been executed in this experiment.

## The bounded baseline implemented here

For every pair of adjacent saved events, compare each group A with each group B.
A sample of A is covered if at least one sample of B is within the displacement
gate and their signed normal angle is at most 60°. Count each covered sample
once, irrespective of how many neighbors it has. Compute the reverse coverage
independently. Save both fractions:

```
parent_coverage = covered samples of A / samples of A
child_coverage  = covered samples of B / samples of B
```

A candidate link passes if both fractions reach the specified threshold. These
fractions are not Jaccard similarity, area fractions or probabilities. Points may
support more than one group pair. There is no exclusivity, conserved ancestry,
velocity prediction, advection, or matching over absent/intermediate events.

The input spatial partitions are normal-compatible graphs with radius 1.5 cells
and angle limits 30° or 60°. Components below 20 samples are excluded from this
association baseline; raw graph labels are retained in the event archives.
Association gates are 1.5, 2.5 and 3.5 cells, with minimum directional coverages
25% or 50%: 12 declared configurations, using all 17 saved events. The displacement
gate is a distance, not a velocity bound; actual time differences are recorded.
Sources, Mach, gas density and known physical shock categories are not inputs.

An event-local group may have multiple candidate parents or children. These
are potential merges/splits, not automatic assertions of physical events.
An unmatched appearance is not proof of shock formation, nor is an unmatched
disappearance proof of shock destruction. Spatial fragmentation, missing
detections, and association thresholds can all produce these patterns.

## Observed results on the current 64³ sequence

With the declared example gate 2.5 cells and minimum coverage 25%:

- Both spatial partitions retain links from the two outer groups to the combined
  group. The receiving event is t=.08 for the 60° partition and t=.10 for 30°.
  This is consistent with the earlier spatial connectivity study, not an
  independent determination of physical collision time.
- The association preserves the two-parent possibility rather than choosing
  one predecessor and silently discarding the other.
- At t=.13 both partitions contain ten 20-sample inner patches in addition to
  the common outer component. At t=.14 those patches become two larger inner
  groups. Several patch-to-group links have only about 19.7% child coverage
  and therefore fail the 25% test despite covering all samples of the patch.
  The raw scores remain saved. This is a limitation of the pairwise coverage
  rule combined with the current spatial segmentation.
- From .14 to .16, the two inner groups have supported links in this example.
- The broad 3.5-cell / 25% setting gives additional multiple-parent candidates
  when patches join. This demonstrates sensitivity, not extra independently
  validated physical mergers. Across all 12 settings the number of links ranges
  from 26 to 38. No single setting is claimed to be optimal.

The outer merge pattern is promising, but the inner-patch example prevents us
from calling this a finished general tracker. No persistent shock IDs or derived
tracked energetics have been added to the simulation output.

## Files and reproduction

`outputs/two_bubbles_associations_n064/` contains:

- `event_*.npz`: original positions/normals and both local partitions, including
  raw labels for small components.
- `nodes.csv`: event-local group sample counts and centroids.
- `pair_scores.csv`: every group-pair score for each gate, including rejected or
  zero-support pairs. Counts are not summed across alternative settings.
- `candidate_states.csv`: parent/child lists and counts for every configuration.
  Group numbers are local to their event; use `(event_index, group)` to identify
  a node, together with the segmentation setting.
- `summary.csv`: link/branch/unmatched counts per configuration.
- `candidate_associations.png`: a declared example, with centroid x on the
  vertical axis. Equal centroids may overlap. Lines are associations, not
  material trajectories; node size reflects sample count, not physical area.
- `metadata.json`, `verification.json`, `SHA256SUMS`: provenance and checks.

```bash
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.check_surface_associations \
  outputs/two_bubbles_iwr_compgpu9_n064/online \
  --output-dir outputs/two_bubbles_associations_n064_new

/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.verify_surface_associations \
  outputs/two_bubbles_associations_n064
```

The runner requires a new output directory. The verifier checks hashes, group
accounting, candidate links/states, and independently enumerates geometric
support using sparse distance matrices for selected merge/fragmentation steps.
Seven synthetic tests cover merges, splits, duplicate neighbor counting, rejected
geometry, row reordering, empty groups and invalid inputs. These verify software
behavior; they do not supply physical truth labels.

## Next step

Inspect the inner patches at .12–.14 and test whether group-level aggregation
or a better spatial surface representation resolves their correspondence.
Keep source ancestry separate from connected-surface identity. A FuzzyCat
same-event ensemble test is possible now; applying it across time requires
an explicit common-object representation whose physical meaning is validated.
