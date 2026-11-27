# Unequal four-bubble evolution: one complete saved-data movie

Open **`four_bubble_evolution.gif`**. It contains all 17 saved observations
over t=0–0.16, at one frame per second (17 seconds per loop). The data come
from `outputs/four_bubbles_unequal_iwr_compgpu2_n064/`; no new simulation was
needed. Selected high-resolution PNG frames accompany the GIF.

## Panels

**Left: current smooth surfaces.** Each color identifies a geometric
continuation segment, using the same colors as the group-history panel. The
camera and unit-domain bounds are fixed. The z ≤ 0.75 cutaway removes upper
caps to expose interior structures; it is a display choice, not missing
simulation data. Black stars and S1–S4 annotations locate the sources and
their strengths; surfaces may occlude them in later frames. Gray points
retain samples outside approved mesh groups. Colors do not assign physical
ownership to the four sources.

**Upper right: current Mach and sampled-density-ratio distributions.** These
are the original saved histogram counts, using the full selected catalog,
including samples hidden by the cutaway and those displayed as points. Each
valid surface sample contributes one count; there is no area weighting.
Bins and count axes are fixed over time. Invalid samples remain accounted
for in the original `history.csv` but do not contribute to histogram bars.
Direct positive density ratios below one remain included.

**Lower right: observed group history.** G00–G24 label continuation segments,
not sources or guaranteed physical shock identities. Nodes and links appear
only after their observations become available. Solid lines show accepted
shared relationships; dotted lines reserve the gate-specific alternative
style. Combinations start new segments. Rings identify current observations.

One dashed orange link, from G08 at t=0.13 to G14 at t=0.14, first appears at
t=0.15. The later observation provides additional conditional support for
that earlier connection. It does not promote the link to an accepted event,
change the segment labels or join their statistics. The final interval's
delayed evidence remains pending because no next observation exists.

## What was verified

- All 17 GIF frames were decoded and matched to their expected time/event
  headers. Playback duration is 17 seconds.
- Histogram counts reconstruct exactly from the original catalog values.
- The table join preserves 71 group observations, 25 continuation segments,
  original sample fingerprints and the callback's saved statistics.
- Mesh fitting uses the existing support restrictions and the per-group
  target: at least 95% of samples within 0.25 grid cell of their own mesh.
  Only the 31-sample group at t=0.11 fails; it remains points. Samples excluded
  by grouping also remain points. All five final groups pass.
- Positions, normals, Mach values, group partitions, segment assignments and
  source-overlap flags remain unchanged. Both original simulation directories
  retain their 180-file input manifests.
- Twelve adapter/history tests and fifteen mesh/encoding tests passed.

Source-overlap flags are advisory and remain visible below the panels.
The final frame records 10,187 samples, five retained groups and 46 upstream
endpoints inside injection regions. Its 42 samples excluded by grouping are
retained as raw points. Those flags still limit physical interpretation of
some measurements.

The meshes are independent display fits, not new physical measurements.
Smoothness does not establish correct topology, resolved nearby sheets in
one group, complete shock detection, physical ancestry or turbulence. No
hole filling or temporal interpolation is added.

`verification.json` records GIF order, causal visibility, fixed axes/view and
input/source/output hashes. `artifact_checks.json` contains the final joins
and integrity audit. Meshes, accuracy checks, environment versions and joined
history tables are in `outputs/four_bubbles_unequal_display_n064/`.

Reproduction commands and method explanations are in
`experiments/wind_bubble/MULTI_SOURCE_ANIMATION.md`. This completes the bounded
four-source visualization step; higher resolution and additional source
parameters are separate experiments.
