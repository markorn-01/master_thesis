# Size-cutoff sensitivity and false-link controls

This CPU experiment tests two unresolved questions from pooled temporal support:
whether the apparent start of inner groups depends on the size cutoff, and
whether high pooled coverage is sufficient evidence for physical identity.
It reads the same 17 saved 64³ catalogs. No hydrodynamics, shock-finding physics,
online output, or production tracking behavior was changed.

## Settings

The raw spatial graphs use radius 1.5 cells and signed-normal angle limits
30°/60°. Minimum group sizes are 1, 8, 16, 20 and 32 samples, applied inclusively.
The raw labels are unchanged; this only changes which components enter matching.
The cutoff-20 partitions are checked against the existing graph implementation.

All 16 adjacent event pairs are compared using displacement gates of 1.5/2.5
cells. Parent and pooled-child coverage thresholds are .50, the best-versus-
runner-up dominance gap is .25, and the association normal limit is 60°.
This gives 320 adjacent-event/configuration comparisons. Thresholds are declared
test settings, not a search for an optimal physical classifier.

The additional diagnostic counts the child samples supported by a candidate
parent **and not covered by its other candidate parents**. Zero exclusive
support flags redundancy. It does not automatically reject the parent or prove
that it is unrelated. In particular, all competing parents can have zero
exclusive support, so deleting every flagged parent would destroy valid evidence.

## Real-sequence results

For the illustrated 60° spatial partition:

| Minimum samples | Groups retained at .12 | At .13 | At .14 |
| --- | ---: | ---: | ---: |
| 1 | 11 | 11 | 11 |
| 8 | 11 | 11 | 3 |
| 16 | 3 | 11 | 3 |
| 20 | 1 | 11 | 3 |
| 32 | 1 | 1 | 3 |

One group is the common outer component; the additional groups include inner
patches and, at cutoff 1, isolated samples. These are graph-component counts,
not counts of independent physical shocks.

At t=.12, the ten inner patches already contain 14–16 samples each. Thus the
20-sample cutoff hides their presence, while cutoff 8 retains them. With the
2.5-cell association gate, cutoff 8 supports all ten .12→.13 patch links and
retains the pooled .13→.14 correspondence. This resolves a cutoff-induced
omission in the candidate graph, not the physical time of shock formation.
The .11→.12 inner appearances remain unmatched because the previous catalog
contains only the common outer component in this spatial representation.

Cutoff 1 also retains eight isolated samples at .14. At .14→.15, these supply
eight accepted links with zero exclusive child support in the 2.5-cell example.
They add no unique coverage beyond other candidate parents. Their acceptance
illustrates why retaining every component is not sufficient to establish a
reliable tracker. It does not establish eight false physical links in the real
simulation: no ground-truth ancestry is available there.

## Controlled false-link example

Construct two parallel 3×3 sheets in the previous event, at z=0 and z=.1,
both with normal +z. The next event contains a sheet at z=0. For this toy
example we stipulate that parent 0 persists and parent 1 is unrelated and
disappears. A .2 distance gate allows both sheets to cover all nine target
samples. Each parent has only one candidate target and passes the dominance
test. Pooled coverage is 100%, yet the rule accepts the stipulated wrong parent.
Both accepted parents have zero exclusive child support.

This is a deliberately ambiguous input, not a claim that geometry can reveal
the stipulated truth. The same snapshots could be consistent with other
physical histories. The result demonstrates that this geometric information
alone cannot establish ancestry. It is not a false-positive-rate measurement
for the hydrodynamic sequence.

The other controls behave as intended:

- Moving the distractor outside the distance gate rejects it.
- Reversing its normal rejects it.
- Two complementary patches that support different parts of a child are both
  retained, with nonzero exclusive support.

## Decision

Keep pooled support as **candidate evidence**, with explicit cutoff settings,
missing links, competing targets and redundancy flags. Do not promote accepted
links to physical shock identities or infer merger energetics from them.
No default size cutoff was changed in the existing pipeline.

The next bounded FuzzyCat experiment can use a single event's common catalog
rows across multiple spatial clusterings. That tests grouping stability under
parameter changes without inventing persistent point IDs. It will not by itself
solve the ancestry ambiguity demonstrated here. Across-time physical tracking
still needs a meaningful common-object representation or additional evidence
such as validated motion prediction or tracers.

## Files and reproduction

Results: `outputs/two_bubbles_support_robustness_n064/`.

- `size_sensitivity.png`: component-count and excluded-sample sensitivity.
- `parallel_sheet_control.png`: the stipulated false-link counterexample.
- `group_sizes.csv`: raw/retained counts for all events and cutoffs.
- `children.csv`, `links.csv`: pooled support and overlap/redundancy evidence.
- `summary.csv`: the 320 comparisons; alternative settings are not independent
  realizations and their counts should not be pooled as physical statistics.
- `event_*.npz`: original positions/normals, raw graphs and cutoff partitions.
- `controlled_examples.json`: expected versus accepted toy correspondences.
- `metadata.json`, `verification.json`, `SHA256SUMS`: provenance and checks.

Readback checks verify input/source hashes, all 170 cutoff partitions, summary
counts, and selected support-overlap counts using independent neighbor
enumeration. Unit tests record the known counterexample as expected behavior;
passing software checks must not be confused with passing that false-link
control. `verification.json` explicitly records the false-link control as failed
and physical identity assignment as not ready.

```bash
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.check_support_robustness \
  outputs/two_bubbles_iwr_compgpu9_n064/online \
  --output-dir outputs/two_bubbles_support_robustness_n064_new

/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.verify_support_robustness \
  outputs/two_bubbles_support_robustness_n064
```

The runner requires a new output directory. No GPU is required.
