# Controlled tests of temporal surface correspondence

## Why this step

The two-bubble simulation has no independent catalog of true shock identities.
A visually plausible link cannot establish correct ancestry. This benchmark
supplies small synthetic surface catalogs with correspondences stipulated when
they are constructed, so false and missed links can be counted explicitly.

This tests the existing pooled geometric rule, not a new tracker. Production
analysis, clustering, Mach reconstruction, GPU runs and earlier outputs are
unchanged. Synthetic group labels are supplied directly: clustering errors are
not included in this test. Truth pairs are used only for evaluation, never as
inputs to geometric matching.

## Setup

Each surface is a 4×4 planar point grid. Some cases contain two sheets. Normals
are +z except for the explicitly opposite-normal control. The three distance
gates .10/.20/.35 use arbitrary toy units, not simulation cell widths. The
baseline is .20. Signed-normal limit 60°, parent and pooled-child coverage
thresholds .5, and parent dominance margin .25 match the earlier rule.

Nine constructed cases are evaluated at all three gates. Independent row
permutations (seed 42) check every configuration without treating row indices as
persistent point identities. Fresh sampling here is a uniform positional offset;
it does not test arbitrary resampling, noise or partial visibility.

| Case | Stipulated correspondence | Result at gate .20 |
| --- | --- | --- |
| Translation .08 | Same sheet persists | Correct |
| Translation .30 | Same sheet persists | Missed: outside gate |
| Fresh sampling | Same sheet, shifted sample positions | Correct |
| Approaching aligned sheets | Two sheets persist separately | Correct |
| Approaching opposite normals | Two sheets persist separately | Correct |
| One group splits into two | Parent corresponds to both halves | Both missed: target tie |
| Two fragments form one group | Both fragments correspond to child | Correct pooled support |
| Nearby sheet disappears | Only parent 0 survives | False link from disappearing parent 1 |
| Distant sheet disappears | Only parent 0 survives | Correct |

Six of nine cases have exact stipulated correspondence at the baseline gate.
This is a description of these chosen examples, not an estimate of accuracy in
hydrodynamic simulations. Split/merge cases describe changes in catalog
representation, not physical shock splitting/merger solutions.

## What the failures tell us

- **Motion beyond a search gate:** the .30 translation is missed at .10/.20 and
  recovered at .35. A larger gate helps this example.
- **Wider is not always better:** the aligned approaching sheets move .12,
  so .10 misses them. At .20 the correct correspondences are recovered. At .35,
  both targets cover each parent equally; the rule reports ambiguity and
  accepts neither. Opposite signed normals remove this competition in the
  corresponding control, but cannot distinguish all real nearby surfaces.
- **The current rule is asymmetric:** one parent must choose a dominant target.
  A balanced one-to-two split produces a tie and is not represented. Pooling
  two fragments into one child succeeds. Motion prediction alone will not
  change this split limitation.
- **Disappearance remains ambiguous:** the nearby disappearing sheet is also
  accepted as a parent of the survivor at every tested gate. Both parent links
  are redundant because they cover the same child samples. Redundancy is an
  ambiguity warning, not sufficient evidence to identify which parent survived.

Do not promote all accepted geometric links into physical identities. No single
distance gate solves these examples. A motion-aware experiment is a reasonable
next step for displacement and nearby-target ambiguity, but cannot be assumed
to resolve identity when the observations themselves are indistinguishable.
Split handling and disappearance need explicit uncertainty/event logic.

## Outputs and verification

`outputs/controlled_tracking_baseline/` contains nine input NPZ catalogs,
`summary.csv`, `results.json`, `controlled_tracking_summary.png`, provenance,
readback and checksums. The figure separates false links from missed links;
zero in one panel alone does not establish successful correspondence.

All 27 configurations reproduce their group-level results after row shuffling.
An independent dense-distance/dot-product calculation, rather than a KD-tree,
reproduces parent choices, child unions, exclusive support and evaluated link
sets for all 27 saved configurations. Seventeen focused tests pass, including
five new tests that preserve the known failures as explicit expectations. A
passing software test does not mean every stipulated correspondence succeeds.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.check_controlled_tracking \
  --output-dir outputs/controlled_tracking_baseline_new

/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.verify_controlled_tracking \
  outputs/controlled_tracking_baseline_new

/opt/anaconda3/envs/astrolink-env/bin/python -m unittest \
  tests.experiments.test_controlled_tracking \
  tests.experiments.test_patch_support \
  tests.experiments.test_support_evidence
```

Use a new output directory. NumPy, SciPy and Matplotlib suffice. No GPU, JAX,
AstroLink or FuzzyCat execution is required. Existing animation work remains a
separate set of changes.
