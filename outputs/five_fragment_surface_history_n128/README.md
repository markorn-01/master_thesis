# Five-fragment history beside the actual surfaces

Open `five_fragment_history.gif`: five saved events from t=0.16 to 0.20,
played at one frame per second. This is a focused review of existing data,
not a new simulation, tracking fit or interpolation of surface motion.
Stills at 0.18, 0.19 and 0.20 are `event_0018.png` through `event_0020.png`.

## Read the three panels

1. Left: actual original refined positions of detected samples near the weaker
   source with inward normals. Black star is the source. Gray samples fall
   below the retained-group cutoff. The source-local inward inspection probe
   matches the earlier review; the outer and stronger-source surfaces are
   omitted only for visibility. Fixed camera and bounds; no clipping,
   smoothing, subsampling or position changes.
2. Middle: original history allowing at most three parents. G12 continues
   from 0.16 to 0.19; G13–G16 appear as retained geometric patches at 0.17 and
   continue through 0.19. At the last interval the wide-gate three-parent
   subset is dashed. The narrow gate has no accepted inner event, so there
   is no shared combination.
3. Right: separate validated experimental history with a five-parent event
   allowed only at the last interval. All five earlier patches connect to
   G17 under both matching distances. All earlier events and segment IDs
   remain identical to the original graph.

Graphs intentionally show the whole saved window at every frame. Black
vertical cursors and black rings indicate the time of the displayed surface.
Future graph points are known saved observations, not online predictions.

Colors consistently identify local continuation segments in surfaces and
both graphs. G17 has its own color because a combination starts a new segment;
it does not inherit a single parent's color or establish material ancestry.
At 0.19 the left view has the five retained parent colors; at 0.20 the later
488-sample group is purple. This is evidence of a supported geometric
relationship, not proof that five physical shocks merged.

The bottom note preserves population and quality counts. At 0.20, 72 of the
488 upstream sample positions overlap an injection region. Associating the
geometry does not validate those samples' Mach or heating reconstruction.

## Checks

The script verifies prior history/experiment input and source hashes, joins
points to nodes by actual event/group IDs, checks exact coordinate equality
and fixed-bound inclusion, confirms unchanged node segment IDs and all events
before the final interval, and checks the shared full-five parent relation.
All five GIF frames decode. Both the final and earlier stills were visually
reviewed. `verification.json` records input hashes, selected-coordinate hashes,
frame counts and explicit limits. No original results were overwritten.

Reproduce using a new output directory:

```sh
python -m experiments.wind_bubble.animate_five_fragment_history \
  --output-dir outputs/five_fragment_surface_history_n128
```

Related quantitative evidence and controlled tests:
`outputs/five_fragment_arity_review_n128/`.
