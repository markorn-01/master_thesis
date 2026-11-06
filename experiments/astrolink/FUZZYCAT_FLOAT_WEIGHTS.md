# Isolated floating-weight correction

## What changed

The unmodified FuzzyCat 0.3.2 helper stores reciprocal multiplicity weights in
an integer array, so a weight such as 1/2 becomes zero. The previous experiment
identified one affected selection among 38 cases.

This experiment defines a local, explicitly experimental subclass. It replaces
only `_setupHierarchyInformation_njit` with a NumPy helper returning floating
reciprocals. Original-index assignments, missing-realization zero weights,
cluster selection and normalization conventions are preserved. It is scoped
to package version 0.3.2. The installed package and base class are not patched.

The subclass is a numerical experiment, not a production shock estimator or
an upstream FuzzyCat release. Its helper is not a performance optimization;
no runtime comparison is claimed.

## Comparison design

For each of the 19 existing same-event ensembles, materialize the archived
cluster files temporarily. Seed both the original class and the experimental
subclass with the recorded cluster-file order and Jaccard graph, then execute
fresh library aggregation and extraction at stability thresholds .50/.75.
Freezing these inputs isolates the weight calculation from file-enumeration
and equal-similarity ordering effects. It is not a fresh AstroLink run or an
independent physical realization.

The runner requires the installed library implementation hash to match the
original recorded experiment. It verifies that fresh base-class execution
reproduces all archived original results, that aggregation/group selection
remains identical, and that corrected memberships agree with the independent
NumPy floating-weight reconstruction. Installed-package and input hashes are
checked again after execution.

## Results

- The minimal reciprocal-weight control now returns .5/.5, as expected.
  Calling the unmodified base class still returns 0/0, confirming isolation.
- All 38 baseline selections were reproduced. Selected group intervals,
  orderings, stability and Jaccard selection scores remain unchanged.
- 37 selections have unchanged memberships within absolute tolerance 1e-12.
- The affected case remains .08, hierarchical input, AstroLink shuffle seed 7,
  minStability .50. Its memberships change for all 3096 samples, with maximum
  absolute flattened-membership difference .1875.

| Affected case | Original | Floating weights |
| --- | ---: | ---: |
| Selected fuzzy groups | 1 | 1 |
| Samples with nonzero maximum membership | 498 | 3096 |
| Samples assigned for display at maximum membership ≥.5 | 296 | 296 |

The restored positive memberships are mostly below the display threshold.
They do not mean that all samples are now confidently assigned to a physical
shock. Membership is determined by the declared parameter ensemble and the
library's normalization convention, not calibrated physical probability.
At this display threshold, labels are unchanged in all 38 selections.

The correction addresses the numerical loss of fractional contributions. It
does not solve contact-stage segmentation, source ancestry, the geometric
false-link counterexample, or the absence of common objects across time.

## Files and verification

Results: `outputs/two_bubbles_fuzzycat_float_weights_n064/`.

- `membership_correction.png`: original, corrected and difference views for
  the affected case, using a thin section of the full 3D results.
- Ensemble directories contain both stability selections, corrected weights,
  memberships, hierarchy and unchanged selected-group information.
- `summary.csv`: all 38 comparisons, including unchanged cases.
- `weight_control.json`: original versus corrected minimal control.
- `execution_checks.json`: fresh base/subclass execution comparisons.
- `verification.json`: saved-array checks against an independent floating
  reconstruction, plus input/source integrity.
- `metadata.json`, `SHA256SUMS`: provenance and output integrity.

Four new unit tests cover fractional/zero weights, original-index mapping,
empty selections and isolation of the subclass from the base class. Existing
mapping, hierarchy-correction and library smoke tests remain applicable.
The saved-array verifier does not require the optional FuzzyCat installation.

```bash
PYTHONPATH=/tmp/thesis-fuzzycat-deps NUMBA_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.check_fuzzycat_float_weights \
  outputs/two_bubbles_fuzzycat_n064 \
  --output-dir outputs/two_bubbles_fuzzycat_float_weights_n064_new

/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.verify_fuzzycat_float_weights \
  outputs/two_bubbles_fuzzycat_float_weights_n064
```

The temporary FuzzyCat 0.3.2 installation from the previous experiment is
required for execution. The output directory must be new. No GPU is needed.

## Decision

The isolated numerical correction behaves as intended on these saved cases.
Keep it explicitly version-scoped and separate from the production pipeline.
FuzzyCat remains useful for same-event grouping robustness, but these tests
do not establish it as a physical temporal tracker. The next investigation
should distinguish data-resolution limitations from clustering choices before
adding more matching rules or interpreting group histories as physical ancestry.
