# Unequal winds: frozen grouping, event tracking and support

Open `unequal_history.gif`: 17 saved events at 2 FPS (8.5 seconds).
Left: all original detected samples. Middle: cutaway hiding z > 0.5.
Right: history graph with the same G labels and synchronized time cursor.
Gray samples are below the retained-group cutoff. Source stars mark positions,
not inferred ownership. G labels represent continuation segments, not material
trajectories or validated physical shock identities.

**Main result:** without changing parameters, the two outer histories combine
at t=0.08→0.09 and the combined group continues to 0.16. The equal-wind 64³
combination was at 0.07→0.08. These are saved-event transitions, not exact
physical collision times. Both gates agree in all 16 unequal intervals.

Only one inner group is retained and followed at the end, near the stronger
source. G03 is a 20-sample patch at 0.13; G04 follows the larger inner group
from 0.14 to 0.16. The predecessor relation is left unlinked: the patch covers
only 16.67% / 25% of the later group, below the fixed 50% family requirement.
This does not establish the absence of another physical reverse shock.

`equal_unequal_comparison.png` / `equal_comparison.json` compare the histories.
`graph.json`, `tracking_results.json.gz`, `fragment_windows.json` and CSV files
retain all decisions, including failed coverage and empty evidence. Raw grouping
files preserve components below the 20-sample cutoff. Original data are intact.

All 32 event optimizations passed independent DP, 50 hypotheses passed sparse
geometry checks, and all 12 eligible family configurations passed dense checks.
All 17 animation frames decode; 27 focused tests passed. Four final inner
samples retain upstream source-overlap flags. See `verification.json` and
`experiments/wind_bubble/UNEQUAL_HISTORY.md` for details and reproduction.
