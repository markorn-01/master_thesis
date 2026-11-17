# Adaptive local implicit shock-surface display

This is the candidate to review first. Open `interactive_comparison.html` in
a WebGL-capable browser to rotate/zoom, compare representations, expose the
interior with a z <= 0.5 cutaway, and overlay the original samples. The viewer
embeds its JavaScript and scene data; it needs no server or external library
download. Browser interaction could not be tested in this session because no
browser was available. PNGs were rendered and visually reviewed; HTML script
syntax and embedded-data consistency were checked separately.

`bubble_comparison.png` shows original points, defect-checked ball pivoting,
the adaptive implicit candidate, the earlier display-normal-fit variant and
support-trimmed Poisson. The top row is full view; the lower row is a display
cutaway. Mesh colors identify supplied geometric groups, not wind ownership
or permanent physical IDs. `control_comparison.png` shows the known geometries.

## Practical result

The candidate constructs a smooth local implicit approximation within each
supplied group, limits it to locally supported samples, and removes detected
mesh defects. It halves extraction spacing only where the original samples
are not adequately represented. No support or normal threshold is relaxed.

For the existing 64³ three-source snapshot at t = 0.16 (7,134 samples):

| Measurement | Original guarded ball pivoting | Ball pivoting after the same cleanup | Adaptive implicit candidate |
| --- | ---: | ---: | ---: |
| Faces | 10,103 | 6,942 | 91,440 |
| Edge-connected pieces | 299 | 131 | 5 |
| Samples within 0.25 cell of triangles | 92.44% | 67.80% | 99.72% |
| 95th-percentile sample-to-triangle distance, cells | 0.401 | 1.643 | 0.143 |
| Nonmanifold vertices | 770 | 0 | 0 |
| Detected self-intersections | No | No | No |

Only the two 177-sample groups need extraction spacing reduced from half a
cell to a quarter cell. The outer group and 226-sample group keep half-cell
spacing. All four meet the per-group target of at least 95% of their samples
within a quarter cell. Twenty samples remain farther than a quarter cell
from the complete display mesh. Five pieces for four supplied groups means
the mesh is not forced into one closed component per group.

The candidate has no detected nonmanifold edges, inconsistent shared-edge
winding or nonmanifold vertices, and no native Open3D self-intersections.
These are numerical checks with library precision, not an exact proof of
physical topology. Sample distance measures fidelity to the detections,
not the simulation's error against a known physical shock.

## Controls and the important failure

The sphere, removed cap and planar-hole controls have no sampled hole-interior
intrusion. The separate-sheet case retains two components on the two known
planes, with 100% quarter-cell sample agreement. Two additional translated
and rotated planes pass the fixed accuracy criterion in the software tests.
These selected controls are not an exhaustive or blinded benchmark.

The deliberately unresolved **two sheets assigned to one group still fail**:
the method creates one intermediate sheet. All samples lie within a quarter
cell, yet all evaluated output probes lie farther than 0.02 cell from either
stipulated true plane. This demonstrates that a good sample-distance score
cannot establish correct topology. Refinement does not recover information
that the input grouping and geometry do not distinguish.

Thus `conditional_display_eligible: true` means this candidate passed the
listed nonambiguous controls and event checks. It does not certify every real
group as a single physical sheet or resolve unknown fragment ancestry.

## Scope and checks

- Original positions, physical normals, Mach values and group labels are
  preserved exactly under `original_*` NPZ keys. Generated display vertices
  are stored separately under `vertices`.
- No detection threshold, physical estimator, shock area/heating output,
  group partition or tracking link changes.
- All source/input hashes, output NPZ arrays and image files were checked.
  Twenty focused tests passed, including the explicit unresolved-sheet
  counterexample, defect removal and translated/rotated boundary checks.
- This uses one real event; it does not validate animation flicker, timing
  cost, higher resolutions or arbitrary turbulent scenes. Grids have a
  three-million-node per-group limit and at most two refinements.
- `cross_nearest_truth_sheet_faces` in the CSV is a nearest-input-label proxy,
  not a physical bridge count for generated vertices. Known-plane offset and
  supplied-group membership are checked independently.

`summary.csv`, `refinements.json` and `verification.json` contain the complete
numbers and provenance. See
`experiments/wind_bubble/SURFACE_DISPLAY_RESEARCH.md` for paper links, equations
and reproduction commands. This remains an optional CPU display pipeline;
existing production defaults and saved scientific measurements are unchanged.
