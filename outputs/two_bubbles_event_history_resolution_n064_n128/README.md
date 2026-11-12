# 64³ versus 128³: automatic group histories

The fixed tracker was applied to the saved 64³ catalogs and compared with the
existing checked 128³ graph. No simulation or scientific parameter was changed.

**Result:** both recover the two early groups combining at t=0.07→0.08 and
the combined outer history continuing to t=0.16. Inner patches first appear at
t=0.13 at 64³ versus 0.10 at 128³; the two later inner histories start at
0.14 versus 0.11. Exact inner-fragment ancestry remains weakly distinguished.

- `resolution_histories.png`: full graphs, with independent segment labels.
- `resolution_summary.png`: descriptive counts, orientation and largest-group
  statistics. Orientation is relative to the nearest source, not ancestry.
- `n064_graph.json`, `n064_results.json.gz`: new graph and complete selections.
- `n064_selected_events.csv`: selections and exclusion objective gaps.
- `group_statistics.csv`, `event_summary.csv`: both resolutions' measurements.
- `n064_distance_agreement.csv`: 16/16 intervals agree between distances.
- `verification.json`: setup checks, independent optimization/geometry checks,
  source/input hashes and result counts.

All 32 new comparisons passed independent optimization checks; geometry checks
covered 94 hypotheses. Nineteen focused tests passed. Inner-combination gaps
at 64³ are only 1.03e-8–4.25e-8, near the 1e-8 numerical tie threshold;
cross-distance agreement is not proof of physical correctness.

Matching distances are fixed in physical coordinates; the existing grouping
radius is fixed in native cells. Detection and grouping therefore contribute
to differences. This is a pipeline sensitivity check, not proof of physical
convergence. See `experiments/wind_bubble/EVENT_HISTORY_RESOLUTIONS.md`.
