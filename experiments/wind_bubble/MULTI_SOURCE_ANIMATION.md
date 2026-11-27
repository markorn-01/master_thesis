# Smooth surfaces, distributions and callback group histories

The next visualization uses the completed unequal four-source run. It adds
display products only: no hydrodynamic evolution, detection, grouping or
association is repeated or changed.

## How to read the movie

- **Left:** the current reconstructed surface, colored by geometric
  continuation segment. Black stars label injection sources `S1`–`S4` and
  their strengths; these are different from the history's `G00`–`G24` segment
  labels. A fixed z ≤ 0.75 cutaway removes the upper cap to reveal interior
  structures. Gray points retain detections outside approved mesh groups.
- **Upper right:** current Mach and direct sampled-density-ratio histograms.
  Both use the original full catalog, including samples hidden by the cutaway
  or displayed as points. Each valid sample contributes one count. Invalid
  samples are excluded from the histograms and remain counted in the original
  `history.csv`; the histograms need not sum to the full detected count.
- **Lower right:** observed group history, using the callback's saved IDs and
  relationships. Solid lines show accepted shared relationships; dotted lines
  identify any gate-specific alternatives. Dashed orange links are additional
  conditional support and appear only when their supporting future observation
  has arrived. Combinations start new continuation segments. A fixed empty
  history axis does not imply that future detections are already known.

Spatial bounds, camera, histogram bins/count limits and segment colors remain
fixed through all 17 frames. At one frame per second the movie lasts 17 seconds;
this is presentation speed, not simulation time. There is no temporal
interpolation or interpolation between source strengths.

## Existing display rules are reused

The builder calls `build_adaptive_sequence.build_event`: same-group oriented
planes are blended into a local implicit field, then masked marching cubes
extracts display triangles. Existing local support restrictions, conservative
defect removal, orientation checks and bounded extraction-grid refinements
remain unchanged. The existing final combined cleanup can remove intersecting
faces after refinement; it cannot add faces or move retained triangle geometry.

At least 95% of each approved group's samples must lie within 0.25 grid cell
of that group's mesh. Failed groups remain original points. The initial empty
state remains empty. The fit is independent at each time, so this does not
establish flicker-free temporal reconstruction, correct topology, or resolution
of very close sheets within one group. No holes are filled to improve appearance.
Mesh areas are not substituted for physical diagnostic measurements.

The table adapter checks original row/position fingerprints, saved group
statistics, segment assignments, both association gates and all four delayed
channels. Conditional support is kept out of statistical continuity. The
terminal interval remains pending when no later observation exists.

## Reproduction

Use a separate CPU display environment with the existing optional requirements
in `surface_display_requirements.txt` (Open3D 0.19.0, scikit-image 0.25.2 and
Matplotlib). The hydrodynamic/JAX environment need not be changed. Run from the
repository root:

```sh
python -m experiments.wind_bubble.animate_multi_source_groups \
  --root outputs/four_bubbles_unequal_iwr_compgpu2_n064 \
  --display-dir outputs/four_bubbles_unequal_display_n064 \
  --build-only

python -m experiments.wind_bubble.animate_multi_source_groups \
  --root outputs/four_bubbles_unequal_iwr_compgpu2_n064 \
  --display-dir outputs/four_bubbles_unequal_display_n064 \
  --output-dir outputs/four_bubbles_unequal_animation_n064 \
  --render-only --fps 1 --cutaway-z 0.75
```

Use fresh output directories to repeat a completed build. `--resume` reuses
completed mesh events only if the input, code and settings hashes are unchanged.
Raw simulation files are never overwritten. GIF verification decodes every
frame and checks the rendered time/event header against the expected sequence.
Selected PNG frames provide static inspection views.

Tests cover source-count adaptation, preservation of callback records,
conditional-link availability, the explicit 25-color palette, existing mesh
support/cleanup/orientation and GIF frame-order checks. Runtime spent making
these CPU display artifacts is not a benchmark of online analysis.
