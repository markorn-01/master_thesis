# Spatial inspection of two-bubble temporal candidates

This stage makes the saved temporal comparison reviewable in space. It does
not rerun hydrodynamics, modify grouping, accept additional links, assign
physical shock IDs, or create a smoothed surface mesh.

Results: `outputs/two_bubbles_spatial_links_n064_n128/`.

## How to read the figures

The first two columns show the previous and next events. The upper panels
project samples near z=.5 onto xy, and the lower panels project samples near
y=.5 onto xz. Both use a physical slab half-width of .04 and fixed spatial
bounds. These are projected point samples, not gas slices or triangulated
surfaces. Gray crosses are size-excluded samples; stars show wind sources.
All clustering and matching used the full 3D catalogs, not these slabs.

The right column is a schematic graph. P/C denote previous/next event-local
group labels, followed by the full 3D sample count. Node colors match their
own event's spatial panels. Colors are assigned independently at each event;
a matching color across columns is not a persistent identity.

- Green: an accepted geometric candidate link under the saved pooled rule.
- Orange dashed: an accepted link whose support adds no exclusive child samples.
- Red dotted/dashed: the best and runner-up targets of an ambiguous parent.
- Red node border: ambiguous parent or child without accepted pooled support.

The lines connect groups in a schematic layout. They do not represent individual
sample trajectories, material motion, or proof of a physical merger. Network
positions are not physical coordinates.

The temporal archive records runner-up coverage but not the runner-up label.
The plotting script reconstructs group-pair coverages using an independent
per-point neighborhood-query helper, verifies the archived best/runner-up scores,
and saves all reconstructed pair scores in `display_audit.json`. Competing
target labels are derived from these queries, not guessed from the display.

## Five views

1. `n064_fragments.png`, .13→.14, native radius/cutoff 20, narrow association
   gate: ten small inner patches collectively support two larger children.
   The graph illustrates a change in the numerical representation; it does
   not establish ten separate physical shocks merging into two.
2. `n128_large_groups.png`, .13→.14, native radius/cutoff 80, narrow gate:
   three retained groups have three accepted candidate links. Smaller groups
   excluded by this cutoff remain visible as gray crosses where they intersect
   the display slabs.
3. `n128_fragments_narrow.png`, same interval, cutoff 20, narrow gate:
   five groups have five accepted links, including two small facing patches.
4. `n128_fragments_wide.png`, same grouping and interval, wider gate:
   the two small parents have competing nearby child targets. Their matches
   become ambiguous; three links remain accepted. A wider search is not
   automatically a better association rule.
5. `n128_redundant_links.png`, .14→.15, cutoff 20, wider gate:
   five accepted links enter three children. Two small-parent links add no
   exclusive child support because other parents already cover those samples.

Spatial normal-angle limit is 60° in every view. Native graph radius is 1.5
cells. Cutoffs 20/80 are sample counts, not measured areas. The two association
gates are 1.5/64 and 2.5/64 code units, with signed-normal limit 60°, parent and
pooled-child coverage thresholds .5, and parent dominance margin .25. No source
ownership or Mach values enter matching. The higher-resolution source budget
and physical comparison limitations remain as documented in
`TWO_BUBBLE_RESOLUTION_RESULTS.md`.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.wind_bubble.plot_resolution_links \
  outputs/two_bubbles_temporal_comparison_n064_n128 \
  --output-dir outputs/two_bubbles_spatial_links_n064_n128_new
```

Requires NumPy, SciPy and Matplotlib; no GPU, JAX or FuzzyCat. Use a new output
directory. Saved input hashes are checked before and after rendering. Node
counts, displayed slab counts, rendered image dimensions and the displayed
candidate links were checked against their saved source records; see
`readback.json`. This is a display-consistency check, not physical tracking
validation. The earlier parallel-sheet false-link control remains unresolved.

These views are suitable for reviewing grouping and matching choices before
building a longer candidate-association animation. Smooth mesh visualization
remains a separate task and should preserve the original diagnostic samples.
