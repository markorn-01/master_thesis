# Focused review of the unequal-wind inner fragments

Open `inner_fragment_review.png`. This is a new view of previously validated
saved outputs, not a new simulation or grouping/tracking run.

Read the top row from left to right:

- t=0.10: seven retained geometric patches (g1–g7), totaling 368 detected
  samples, arranged around the stronger source. Each patch has its own color.
- t=0.11: one substantially more complete inward-oriented group, 1,064 samples.
- t=0.12: its shared continuation, 1,432 samples. Purple denotes the later
  continuation segment; it does not assign ancestry to the earlier patches.

The outer surface is omitted only in this close-up. All selected samples are
shown at their original refined positions, with fixed bounds and camera.
The black star is the stronger source. Group numbers are local to each event;
g1 at different times is not automatically the same physical object.

Read the bottom row from left to right:

A. The actual adjacent-frame decision. Narrow distance: no inner event.
Wide distance: a three-parent subset g2/g3/g6, with a tiny exclusion gap of
3.52e-8. Orange dashed links are this distance-specific alternative. The main
optimizer permits at most three parents, so it cannot select all seven in
one event. The later 0.11–0.12 continuation is shared by both distances.

B. The separate conditional family test, seeded by that later continuation.
All seven earlier patches jointly support the later group under all four
settings (two distances, static and predicted). Dotted lines denote this
conditional support, not new accepted edges in the main tracking graph.

C. Coverage of the later group by the earlier support footprints: 92.48% and
100% at 128³, versus 16.67% and 25% in the previously tested 64³ window.
The fixed acceptance threshold is 50%. Coverage counts later samples within
the normal-compatible matching distance of earlier points (or backward-
predicted points); it is not the ratio of the sample counts 368/1,064.
Static and predicted variants give identical coverage and parent membership.
The 64³ and 128³ bars concern different detected time windows, not matched
states or proof of physical/algorithmic convergence.

Interpretation: the spatial arrangement is consistent with incomplete
patch detection of an inner surface followed by more complete detection.
It does not establish seven independent physical shocks merging. Higher
resolution supplies sufficient geometric support under the frozen rule;
physical ancestry and two complete inner-shock histories remain unproven.

Quality: upstream sampling endpoints overlap the injection region for
112/368 samples at t=0.10, 8/1,064 at 0.11 and 0/1,432 at 0.12.
No downstream overlap occurs in these selected populations. Those flags
concern state reconstruction and are preserved; visual geometric support
alone does not validate the corresponding early Mach/heating estimates.

The review checks prior validation hashes, all four family memberships,
the two actual adjacent decisions, the later shared continuation, exact
sample selections/counts and source-overlap counts. `verification.json`
records input hashes and point-coordinate hashes. No points/normals or
original graph/results were modified.

Reproduce with a new output directory:

```sh
python -m experiments.wind_bubble.review_unequal_fragments \
  --output-dir outputs/two_bubbles_unequal_fragment_review_n128
```
