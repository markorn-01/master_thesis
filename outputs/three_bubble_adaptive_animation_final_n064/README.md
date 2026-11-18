# Corrected full three-bubble surface animation

Open `adaptive_surface_evolution.gif`: seventeen saved events over simulation
t = 0 to 0.16, played at 1 frame per second (17 seconds).

Left: original detected samples. Right: approved adaptive triangle surfaces,
with raw points retained outside approved groups. Both panels use the same
z <= 0.5 cutaway, fixed camera and spatial bounds. Colors identify geometric
continuation segments; black stars mark wind sources. Every event is fitted
independently, without temporal smoothing.

The final combined cleanup fixes the previous event-5 intersection by removing
35 conflicting faces. All sixteen populated events now pass the existing
display criteria; the empty initial state is preserved. Small unsupported
holes are kept rather than filled. The physical samples, measurements and
tracking decisions are unchanged.

`event_0005_decoded.png` is extracted from GIF frame 5 for inspection. Other
event PNGs are standalone renders. `display_checks_over_time.png` summarizes
sample agreement and approved group counts. `verification.json` checks input,
source and output hashes plus decoded GIF frame ordering. `qa.json` records
461 successful hash checks, sixteen unchanged parent meshes, exact original
event-5 arrays, 17 decoded frames of 1000 ms each, and three passing targeted
tests. The local repair comparison, decoded frame 5 and final event were
visually inspected.

The original movie is preserved in `../three_bubble_adaptive_animation_n064/`.
This corrects the remaining detected display intersection in this sequence;
it does not establish physical shock topology, solve uncertain ancestry,
validate temporal flicker or guarantee reconstruction of nearby sheets
assigned to the same group.
