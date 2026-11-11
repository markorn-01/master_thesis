# First event-aware tracking comparison

Start with `aggregate.csv` or `event_tracking_comparison.png`.

On the existing 27 synthetic configurations, exact correspondence improves
from 16 to 23; false links decrease from 3 to 0 and misses from 14 to 6.
The nine additional stress configurations remain difficult: the prototype
has 3 false and 9 missed links, versus 6 false and no misses for the pooled
baseline. These failures are retained, not removed from the report.

`results.json` stores generated hypotheses, solver choices, unresolved
alternatives and truth comparisons. Truth is evaluation metadata, not
tracker input. Input NPZs contain the original catalog arrays. `summary.csv`
has one row per method/configuration. `verification.json` records independent
dense/exhaustive checks, shuffled-row checks, fixed parameters and source
hashes. Read `experiments/astrolink/EVENT_TRACKING.md` for decision rules,
the objective, limits and reproduction commands.

This is an original one-stage conflict-optimization prototype, informed by
feature-event tracking literature. It is not a reproduction of the published
two-step algorithm, not yet applied to the real bubbles, and not validated
physical ancestry. Existing simulation, matching and visualization outputs
are unchanged.
