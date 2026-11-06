# FuzzyCat on common-row, same-event ensembles

## Question and scope

Can FuzzyCat identify groups that recur when AstroLink's parameter choices
change on one saved shock catalog? This test uses common original catalog row
indices, so memberships refer to the same samples within each ensemble.
It does not identify samples across time or assign physical shock ancestry.

FuzzyCat's [API](https://fuzzycat.readthedocs.io/en/latest/api.html) accepts
cluster memberships from representations of the same object set and compares
them through Jaccard similarity. Its [algorithm description](https://fuzzycat.readthedocs.io/en/latest/howitworks.html)
explains how these clusters are combined into fuzzy groups. Here, a realization
means a parameter setting on the same saved data, not a new simulation or an
independent physical/noise realization.

## Inputs

- Three saved events: t=.04 (separated), .08 (contact), .16 (later interaction).
- Each ensemble has 12 declared AstroLink settings: normal weights 0/1/4,
  density neighborhoods 20/40, and significance settings auto/2.
- Original point order and shuffled orders 42/7 form separate 12-setting
  ensembles. They are not combined into 36 independent physical realizations.
- Two encodings: disjoint deepest-non-root display partitions, and the full
  nested non-root AstroLink hierarchy. Original catalog row indices undo the
  earlier input shuffles. No failed AstroLink fits occur in these selected inputs;
  the exporter rejects them if present.
- Realizations without non-root groups contribute no cluster file but remain
  in `nSamples=12`. Dropping them would change the stability denominator.
- The all-points AstroLink root is excluded from the primary experiment. One
  separate contact-stage hierarchy ensemble includes it as a diagnostic control.

Nested memberships retain their hierarchy IDs. Flat input uses sibling IDs
without claiming the same hierarchy. Input clusters are archived as compressed
bundles and temporarily materialized in FuzzyCat's required `.npy` layout.
Source coordinates, Mach, gas density and physical shock labels are not inputs.

## FuzzyCat settings

Unmodified `fuzzycategories==0.3.2`, installed only in a temporary directory;
NumPy/Numba versions and the library implementation hash are recorded. There
are 19 ensembles: 18 primary combinations plus the root control. Each is selected
at `minStability=.50` and `.75`, giving 38 selections. Other settings are
`minIntraJaccardIndex=.50`, `maxInterJaccardIndex=.50`, `windowSize=None`.
Numba uses one thread in the recorded run. No GPU or hydrodynamic rerun is used.

Stability records the fraction of ensemble realizations represented in a fuzzy
group. It is not calibrated physical confidence. Parameter choices can yield
identical clusters and are correlated; the declared weighting of those choices
affects the ensemble. Counts should not be interpreted as repeated independent
experiments.

## Results

The two input encodings give the following fuzzy-group counts. Ranges cover
the three point-order ensembles, not statistical error bars.

| Event | minStability .50 | minStability .75 |
| --- | ---: | ---: |
| t=.04 | 2 | 0–2 |
| t=.08 | 1 | 0 |
| t=.16 | 3 | 3 |

At .04, the original-order groups each have stability 10/12; shuffled inputs
give 8/12, explaining the loss at the .75 threshold. This reflects changes in
the underlying AstroLink clusterings, not the physical appearance/disappearance
of shocks.

At .08, selecting one fuzzy group does **not** recover the whole connected outer
surface. The original-order unmodified results have nonzero membership for
1796 of 3096 samples. At the illustrative max-membership ≥.5 display threshold,
1309 samples are assigned for flat input and 1696 for hierarchical input.
The differences are real differences in those returned membership arrays,
not an assertion that one encoding is physically correct. Contact-stage
memberships vary across the shuffled ensembles. One shuffled hierarchy result
is additionally affected by the implementation issue below.

At .16, both encodings and all point-order ensembles select three groups at
both tested stability thresholds. The original-order display separates 5608
outer samples and two groups of 280 inner samples. This agrees with earlier
grouping, but it is not independent physical validation because the same
AstroLink memberships are being combined.

In the root-included .08 control, FuzzyCat selects an all-points group with
stability 1 and membership 1 for every sample at both thresholds. The root
exists in all inputs by construction. This is a trivial stable population,
not proof that the shocks have been separated correctly.

## Implementation issue found in the tested release

In the installed 0.3.2 implementation, `_setupHierarchyInformation_njit`
allocates `sampleWeights` with integer dtype and assigns reciprocal cluster
multiplicities to it. A value such as 1/2 therefore becomes 0. The minimal
`weight_reproducer.json` records two clusters per realization contributing to
one selected fuzzy group: expected weights [.5,.5], returned weights [0,0].
This numerical control fails.

The audit detects this in **one of 38 selections**: .08, hierarchy input,
shuffle seed 7, minStability .50, with four truncated sample/group weights.
A separate NumPy reconstruction using floating reciprocals changes individual
flattened memberships by up to .1875 in absolute terms. Its selected
clusters are held fixed. This is a counterfactual membership diagnostic, not a
modified FuzzyCat run or a validated physical estimator. The original library
outputs remain unchanged alongside it.

The issue occurs after cluster selection, so it does not alter the reported
selected-group counts. It does affect membership interpretation for that
selection. The original-order figure has no truncated reciprocal weights;
the caption nevertheless flags the issue found elsewhere in the study.
No statement is made about untested development versions of FuzzyCat.

## Outputs and verification

Results: `outputs/two_bubbles_fuzzycat_n064/`.

- `fuzzy_group_counts.png`: selection counts and point-order ranges.
- `contact_memberships.png`: returned contact-stage memberships, plus root control.
- `summary.csv`: all 38 selections, coverage and weight-truncation flags.
- `catalog_*.npz`: saved geometry used only for display and common row context.
- Ensemble directories: `realizations.json`, `input_clusters.npz`, returned
  cluster ordering, Jaccard pairs, group properties, memberships, hierarchy,
  sample multiplicities and library/reference weights in `result_*.npz`.
- `float_weight_reference_*.npz` exists only for an affected selection; it is
  generated by the verifier and does not replace the original result.
- `weight_reproducer.json`, `metadata.json`, `verification.json`, `SHA256SUMS`.

Display labels use maximum flattened membership ≥.5; tied maxima are marked
separately, not resolved arbitrarily. Full memberships are preserved. Colors
are event-local and are not physical identities. Thin spatial sections only
display the memberships computed over all 3D samples.

Verification reconstructs input memberships from the saved AstroLink mappings,
checks all 40,184 recorded Jaccard pairs, reconstructs output memberships and
hierarchy accumulation from saved library weights, and independently checks
stability denominators. A repeated-disjoint-partition smoke test exercises the
actual library. Mapping and numerical checks passing does not erase the failed
reciprocal-weight control or validate the shock physics.

## Reproduce

Using the existing AstroLink environment, install the optional release in a
temporary location without altering the environment's packages:

```bash
/opt/anaconda3/envs/astrolink-env/bin/python -m pip install \
  --no-deps --target /tmp/thesis-fuzzycat-deps fuzzycategories==0.3.2

PYTHONPATH=/tmp/thesis-fuzzycat-deps NUMBA_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.check_fuzzycat_ensemble \
  outputs/two_bubbles_contact_sensitivity_n064 \
  --output-dir outputs/two_bubbles_fuzzycat_n064_new

/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.verify_fuzzycat_ensemble \
  outputs/two_bubbles_fuzzycat_n064
```

The temporary installation assumes the existing environment already provides
NumPy and Numba. The verifier does not require FuzzyCat. The runner requires a
new output directory. No upstream package files are changed.

## Decision

FuzzyCat is callable on our common-row catalogs and provides a useful view of
ensemble stability. This first test does not solve contact-stage segmentation
or temporal physical identity. Before using membership magnitudes, resolve and
retest the reciprocal-weight issue in an explicit isolated comparison. A later
across-time experiment still needs a meaningful common-object representation;
current row IDs must not be reused as temporal identities.
