# Experimental floating-weight FuzzyCat comparison

A local subclass corrects reciprocal-weight truncation without editing the
installed FuzzyCat 0.3.2 package. It re-executes aggregation and extraction on
the frozen recorded Jaccard graphs for all 38 baseline selections.

See the [method, results and reproduction commands](../../experiments/astrolink/FUZZYCAT_FLOAT_WEIGHTS.md).

All selected groups remain unchanged. One selection has corrected memberships:
nonzero support rises from 498 to 3096 samples, but the 0.5 display assignment
count remains 296. The maximum membership difference is .1875.

The numerical control passes with floating weights. This does not establish
physical shock identities or temporal ancestry, and no production integration
was made. Start with `membership_correction.png` and `summary.csv`.
