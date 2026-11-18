# Checked adaptive display over the full saved sequence

`build_adaptive_sequence.py` applies the already checked fitting, support,
defect-removal, refinement and winding stages to all 17 saved three-source
events. `animate_adaptive_sequence.py` renders the checked results beside the
original points. This is optional CPU postprocessing, with no new hydrodynamic
run or change to physical diagnostics, grouping or temporal association.

## Construction and reuse

The fitting parameters remain those in `DisplaySettings`: half-cell initial
extraction, fixed local support and normal criteria, and at most two extraction
refinements. Each retained group must have at least 95% of its original samples
within a quarter cell of its own group's triangles. Combined meshes must pass
native intersection/manifold checks and independent shared-edge winding checks.
The orientation pass can reverse triangle order only; it changes no fitted
surface vertex or triangle membership.

Events 12, 13 and 14 reuse the exact checked oriented candidates from the
three-state validation. Event 16 reuses the checked static adaptive candidate.
Their vertices, physical arrays, group labels, time and complete candidate face
arrays must match the references. Every reused candidate is remeasured. Other
events are constructed independently from their saved catalogs. No information
from another time is used in the fit. The saved geometric history supplies
colors, not spatial reconstruction or wind ownership.

The initial event has no detected samples. It is recorded as `no_detections`,
with undefined distance statistics, and is retained as an empty animation frame
with source markers. It is not counted as a successful measured shock surface.
No-group events and known geometry/resource failures have separate statuses.

## Explicit fallback

Groups below the accuracy target remain raw points. A global geometry or
orientation failure suppresses mesh display for that event. Known bounded
resource/cleanup failures likewise retain the event as points; invalid inputs
or programming errors stop the run. No support or accuracy limit is relaxed to
produce a smooth frame. All original samples remain available in each NPZ.

`candidate_faces` records the candidate geometry; `faces` records approved
display faces. `summary.csv` candidate metrics are distinct from approved
face counts and fallback populations. Points outside retained spatial groups
are also retained as gray fallback points. Group and piece counts are not
physical shock counts. Source-overlap flags remain unfiltered measurements.

## Resume and verification

An event is marked completed only after NPZ readback and its event report are
written. `--resume` checks the same configuration, software-source hashes,
reference hashes, completed-event inputs and output hashes before reuse.
Completed meshes are not silently rebuilt under changed source/configuration.

The complete build verifies source/input hashes again and records per-event
checks plus output hashes. The renderer requires a complete checked build,
checks original physical arrays against the catalogs, and decodes every GIF
frame. Each decoded header is matched against all expected rendered headers to
check event order despite palette quantization. Durations, frame count and
playback time are recorded. Two tests verify correct header-order acceptance
and rejection of reordered expected frames. An empty/first-populated-state
smoke run and explicit resume run also pass.

The movie uses fixed camera, spatial bounds, continuation-segment colors and
the same z <= 0.5 cutaway in both panels. A geometric combination receives a
new segment label/color. Face colors are group/segment colors, not Mach values
or physical source ownership. No intermediate times or temporal smoothing are
added. Source stars remain visible for orientation.

## Run

Use the optional environment in `surface_display_requirements.txt`, separate
from the GPU/JAX environment:

```sh
python -m experiments.wind_bubble.build_adaptive_sequence \
  --output-dir outputs/three_bubble_adaptive_sequence_n064

# Only if the same construction was interrupted:
python -m experiments.wind_bubble.build_adaptive_sequence \
  --output-dir outputs/three_bubble_adaptive_sequence_n064 --resume

python -m experiments.wind_bubble.animate_adaptive_sequence \
  --directory outputs/three_bubble_adaptive_sequence_n064 \
  --output-dir outputs/three_bubble_adaptive_animation_n064 \
  --fps 1

python -m unittest tests.experiments.test_adaptive_animation_encoding
```

Use fresh output directories unless explicitly resuming a build. The result
READMEs report actual outcomes and any fallback events. The movie supports
visual review; it does not establish correct physical topology or material
ancestry, remove the same-group close-sheet ambiguity, or quantify temporal
flicker. No runtime/memory scaling claim is made for this display pipeline.
