# Same-event FuzzyCat experiment

Unmodified FuzzyCat 0.3.2 on saved 64³ two-bubble catalogs: 19 ensembles, each
with 12 AstroLink parameter settings; two stability thresholds give 38 selections.
No hydrodynamic rerun, GPU work, physical IDs or temporal correspondence.

See the [method, findings, numerical issue and reproduction commands](../../experiments/astrolink/FUZZYCAT_ENSEMBLE.md).

- Late t=.16 inputs consistently select three groups.
- At contact t=.08, a lower threshold selects a partial group; a higher threshold
  selects none. Root inclusion produces a trivial all-points group.
- A reciprocal-weight control fails in release 0.3.2. One selection contains
  truncated weights; the unmodified output and a floating-weight reference are
  both retained. This issue is explicitly flagged in `verification.json`.

Start with `fuzzy_group_counts.png` and `contact_memberships.png`. These figures
show grouping stability and returned memberships, not physical shock identities
or calibrated physical probabilities.
