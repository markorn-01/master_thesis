# Group histories on the original shock surfaces

Open `event_history_surfaces.gif` (17 frames, 2 FPS, 8.5 seconds).
Static examples: `event_0008.png` and `event_0011.png`.

1. **Left:** full 3D detected shock sample cloud.
2. **Middle:** the same cloud with z > 0.5 hidden to expose inner groups.
3. **Right:** complete history graph; black cursor/circles mark the current
   time and groups. Matching colors refer to the same G labels in all panels.

The two early groups G00/G01 connect to G02 at t=0.07→0.08. G02 follows the
combined outer surface. Later G17/G18 follow the two inner groups from t=0.11
onward. Dashed connections into them are wide-distance-only alternatives,
not adopted ancestry. Color denotes continuation segment, not shock strength.
Gray points are samples outside retained groups. Stars show wind sources.

All original samples are shown in the full view, with fixed camera and bounds;
no smoothing, triangles, coordinate changes or subsampling. Only visibility
changes in the cutaway. Printed counts always refer to full groups. The graph
labels identify detected-group histories, not established physical shock IDs.

`verification.json` records all 53 group joins, original coordinate hashes,
input/source hashes, display settings and GIF frame checks. Seventeen focused
tests passed. See `experiments/wind_bubble/EVENT_HISTORY.md` for commands and
limitations. No GPU or new hydrodynamic run was required.
