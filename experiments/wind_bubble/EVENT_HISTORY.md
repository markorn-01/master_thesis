# Automatic multi-time group history

This step assembles the frozen event-tracker decisions into one graph. It does
not rerun hydrodynamics, regroup samples, or change matching parameters.

Each node is one detected group at one saved time. Shared one-to-one
continuations inherit a graph segment label. Shared combinations and
separations connect parent and child nodes explicitly but start new child
segments. They do not arbitrarily inherit a parent's identity. Events selected
at only one distance remain alternative hyperedges; they do not propagate IDs.
An unrelated shared continuation can proceed through the same uncertain interval.

## Result for the 128³ two-bubble sequence

- All **53 group observations** across 17 saved times are retained.
- **32 shared continuations** form **21 continuation segments**, including
  isolated observations. Segment count is not the number of physical shocks.
- One shared combination connects the two early histories to the combined
  outer group at **t=0.07→0.08**.
- Two wide-distance-only combinations at **t=0.10→0.11** remain dashed
  alternatives. Their specific parent subsets are weakly distinguished in
  the objective (see `EVENT_TRACKING_APPLICATION.md`). The two later inner
  histories continue from t=0.11 through t=0.16 without choosing these ancestors.
- Two small patch histories exist only at t=0.13 and 0.14. An unlinked end
  does not establish physical disappearance.

`event_history.png` shows every observation and event, including unmatched
groups. Diamonds are branching event junctions; labels beside points are the
original per-catalog group numbers. Vertical G labels identify continuation
segments only. The shaded band marks the distance-sensitive interval.

`segment_statistics.png` shows median valid Mach number and sample counts for
segments containing at least three observations. The symmetric early curves
overlap. Medians use finite positive Mach values with `mach_valid`; counts are
surface sample counts, not area. No curves are interpolated across branches or
gaps. All short segments remain in `group_histories.csv` and `graph.json`.

## Verification and use

The builder verifies the validated application's source and input hashes before
using its archive. Saved graph readback checks complete observation accounting,
adjacent-time edges, continuation inheritance and noninheritance through
branches/alternatives. Six focused synthetic tests cover combinations,
separations, alternatives alongside independent continuations, empty catalogs,
conflicting events and invalid times. The existing seven segment tests also pass.

```sh
python -m unittest tests.experiments.test_event_history tests.experiments.test_continuation_segments
python -m experiments.wind_bubble.build_event_history --output-dir outputs/two_bubbles_event_history_v1_n128
```

Use a new output directory. Outputs include the graph JSON, node statistics CSV,
two PNG figures and hash/verification report. No GPU is needed. Original
provisional-history outputs remain unchanged. This is an automatic detected-group
history, not independently validated physical ancestry or a general turbulent-box
tracker. The next useful check is to review these same segment labels on the
actual shock surfaces, rather than tuning the matching to make a cleaner graph.

## Spatial review animation

`animate_event_history.py` now displays the same G labels on the original
refined 3D surface samples, synchronized with the graph. It uses two spatial
views: every sample in the full view, and samples with z <= 0.5 in the cutaway.
The cutaway reveals inner groups hidden by the outer surface. It changes only
visibility, not grouping, matching, geometry or measurements. Unretained
samples remain gray. Fixed colors identify segments, not Mach numbers.

The muted outer G02 group follows the combined outer surface; G17 and G18
show the later two inner groups. Full group counts are printed below the views.
Small fragments remain visible and recorded. No surface points are tracked
individually; the algorithm relates groups between analysis events.

```sh
python -m unittest tests.experiments.test_event_history_surfaces tests.experiments.test_event_history tests.experiments.test_continuation_segments
python -m experiments.wind_bubble.animate_event_history --output-dir outputs/two_bubbles_event_history_surfaces_v1_n128 --fps 2
```

Seventeen focused tests pass, including four display tests for label joins,
empty/unretained populations, mismatched groups/counts and the cutaway boundary.
All 53 group observations are joined back to original catalog row indices and
coordinates are hashed. The GIF is decoded to verify all 17 frames at 2 FPS
(8.5 seconds). Example PNGs show t=0.08 and 0.11. These checks validate the
display/data correspondence; they do not establish physical ancestry.
