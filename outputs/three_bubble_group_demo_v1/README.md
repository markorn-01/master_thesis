# Three-bubble evolution: surfaces, tracking and statistics together

Open `three_bubble_group_evolution.gif`. It shows 17 saved events over
simulation t = 0 to 0.16, played at 1 FPS (17 seconds). This is offline
grouping/rendering of saved online diagnostics; no new GPU run or tracking
algorithm is introduced.

## Read the four panels

- **Upper left:** current checked smooth shock-surface display. The fixed
  z <= 0.5 cutaway exposes the interior. Black stars are the wind sources;
  gray samples are outside approved mesh groups. The camera and spatial
  bounds stay fixed. Previously verified meshes are reused without refitting.
- **Upper right:** group history available at the current event. Solid teal
  connections are shared accepted geometric events. Gray dotted connections
  are distance-specific alternatives. Orange dashes show conditional support.
  Circles mark observations; black rings mark current groups. G labels are
  continuation segments, not physical source ownership.
- **Lower left:** per-group median Mach and the p16–p84 spread of valid
  original sample values. Bands show sample spread, not uncertainty intervals.
  Symmetric group histories can overlap. Missing/absent observations break
  curves; each combination starts a new segment rather than joining curves.
- **Lower right:** original sample counts per group on a logarithmic scale.
  These are neither surface areas nor display-triangle counts. Only observed
  measurements up to the current event are drawn.

Colors match the same geometric segment across all four panels. The current
group legend changes with the scene. Statistics use each complete group,
including samples hidden by the display cutaway. Conditional support never
merges statistics or assigns an inherited identity.

## Watch the one-event delay

Compare `event_0014.png` and `event_0015.png`:

1. At event 14 (t=0.14), the later B group exists, but the added conditional
   connection is still hidden.
2. At event 15 (t approximately 0.15), C becomes available. The orange dashed
   relationship from historical G08 to G13 appears. The original four-parent
   combination remains solid and unchanged.
3. G08's earlier Mach/count observations remain separate from G13's curves.
   The extra relationship is evidence, not a new physical ancestry identity.

The two gray distance-specific alternatives remain explicit. Source-overlap
counts are retained in the event note; additional geometric support does not
validate source-affected Mach/heating sampling. Smooth display geometry does
not establish physical topology or remove earlier reconstruction limitations.

## Data and checks

The movie reads the verified table in `../three_bubble_group_history_table_v1/`:
45 group observations across 17 events. Sample counts and original coordinate
hashes match the mesh catalogs at every event, and all segment assignments
match the existing history. No samples or mesh vertices are moved.

All 17 GIF frames are decoded and their headers checked against their expected
event. Playback is 17 seconds. Visibility is checked at every event; no future
measurement or relationship is displayed. Empty initial state and fixed axes
are retained. Stills show events 0, 8, 13, 14, 15 and 16. The empty and C-event
renders were visually reviewed, including the graph legend placement.

The exercised focused tests cover five new table/visibility tests, four
existing spatial-review tests, twelve delayed-support tests, two GIF-encoding
tests and six history tests (29 distinct tests, all passing). `verification.json`
and `qa.json` record readback, provenance and checks.

```sh
python -m experiments.wind_bubble.animate_group_history \
  --output-dir outputs/three_bubble_group_demo_v1 --fps 1
```

Use a fresh output directory. Rendering uses CPU postprocessing and does not
establish a general online many-bubble tracker or a new cost benchmark.
