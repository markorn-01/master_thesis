# Surface reconstruction for shock visualization

This work concerns the display of existing detected shock samples. It does
not change detection, physical state sampling, Mach reconstruction, shock-area
or heating estimates, spatial grouping, or temporal association.

## Research and method choice

There is no unique surface implied by an incomplete point cloud. The
[CGAL reconstruction tutorial](https://doc.cgal.org/latest/Manual/tuto_reconstruction.html)
explains the need for assumptions and compares reconstruction families.
In particular, missing detections must not automatically become a closed
surface, and a geometric group can contain unresolved nearby sheets.

| Family | Relevance and treatment here |
| --- | --- |
| Ball pivoting | Reuses the existing original-vertex, single-radius baseline with its fixed geometric guards. The [IPOL implementation paper](https://www.ipol.im/pub/art/2014/81/) describes the ball construction; this is not a newly invented reconstruction method. |
| Screened Poisson | Fits a smooth implicit surface to oriented samples. The [authors' paper](https://hhoppe.com/proj/screenedpoisson/) motivates its use. The [Open3D tutorial](https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html#poisson-surface-reconstruction) explicitly documents extrapolation in low-support regions. Both raw and locally trimmed output are tested. |
| Moving least squares / point-set surfaces | Local approximation is a natural candidate for an open sampled surface. [Alexa et al.'s point-set surfaces](https://www.cs.jhu.edu/~misha/Fall05/Papers/alexa01.pdf) provide the broader motivation. This code implements a simple compact blend of oriented tangent planes, not their complete projection algorithm. |
| Advancing front | [CGAL's method](https://doc.cgal.org/latest/Advancing_front_surface_reconstruction/index.html) interpolates points and produces oriented manifold surfaces with boundaries. Reviewed, not installed or benchmarked here; it offers a different connectivity construction but does not alone smooth the original coordinates or resolve ambiguous input sheets. |
| Scale space | CGAL smooths a point set for meshing and propagates connectivity back to original samples. Reviewed, not benchmarked here. This does not establish a uniquely correct shock surface. |
| Surface splatting | [Zwicker et al.](https://vcg.seas.harvard.edu/content/3-publications/20010101-surface-splatting/p371-zwicker.pdf) render surface appearance without connectivity. Useful as a fallback display family; this experiment does not implement their EWA renderer. |
| Alpha shapes / marching cubes | Alpha shapes are a different geometric prior, not a way to infer shock ownership. Marching cubes extracts an existing scalar field; it cannot reconstruct a point cloud until such a field is defined. Here it extracts the locally fitted field. |

## The new local implicit candidate

For display query x, original sample positions p_i and supplied oriented
normals n_i define

    f(x) = sum_i w_i(x) n_i · (x - p_i) / sum_i w_i(x).

The display surface is the local zero set of f. Weights combine a Gaussian
kernel with a compact Wendland-shaped window. This is a weighted local-plane
blend; it does not integrate the full hydrodynamic jump equations.

Each supplied group is processed independently. Two versions use either saved
physical normals or display-only covariance-plane normals fitted within two
grid spacings. Failed fits retain the saved normal and are counted. No source
position, assumed spherical surface, or temporal link determines the fit.

The frozen exploratory settings are half-cell extraction spacing, two-cell
compact support, 0.8-cell Gaussian width, at least three signed-normal-compatible
neighbors (60 degrees), maximum nearest-sample distance 0.85 cells and maximum
nearest-plane offset 0.30 cells. These are display parameters, not validated
physical uncertainty bounds. Changing them requires another comparison.

Unsupported grid nodes have no valid sign. Extraction requires supported grid
cells, then tests each triangle's vertices, edge midpoints and centroid against
the sample-distance and plane-offset bounds. A local tangent-plane convex-hull
test additionally requires surrounding sample support, discouraging boundary
extrapolation. These seven probes do not prove continuous support over every
triangle. Group bounding grids have an explicit three-million-node limit.

Poisson uses Open3D 0.19.0, depth 6, scale 1.1, linear fitting and one thread.
Its supported variant applies the same local trimming checks, instead of
interpreting Poisson's entire extrapolated surface as a detected shock.

All implicit/Poisson vertices are new display vertices. The original samples
and physical arrays are stored separately and checked exactly. A nearest-sample
index in the research output is an attribution aid, not a measurement at the
new location. No mesh-derived physical area or thermalization is reported.

## Controls, cleanup and acceptance limits

The comparison uses the existing sphere, removed spherical cap, planar hole,
two separately labeled close sheets and deliberately unresolved same-group
close sheets. True sheet labels are evaluation-only. Each algorithm is applied
before inspecting the real three-source event. The method suite and controls
are selected experiments, not a comprehensive or blinded benchmark.

Metrics include original-sample to triangle distances, analytic area checks
on the sphere/cap, sampled hole intrusion, edge-connected pieces, boundary
length, edge winding, nonmanifold vertices/edges and native self-intersection
detection. Area is independently recalculated with Heron's formula. The
known-sheet tests also measure distance to the actual two planes: nearest
input-point distance alone can miss an invented middle sheet.

`finalize_surface_display.py` conservatively removes every triangle incident
to a reported nonmanifold vertex or intersecting triangle pair, repeating up
to eight passes. It fills no holes, relocates no vertices and adds no faces.
The retained oriented triangle coordinates are checked exactly against the
research mesh. Geometry and coverage are measured again after removal.

Conditional eligibility requires no measured topology defects or sampled
known-hole intrusions, at least 95% of samples within a quarter cell of the
mesh, and less than 1% off-plane probes for the separately labeled sheet
control. These exploratory acceptance cutoffs are not universal tolerances.
The same-group counterexample remains an explicit failure condition for a
universal reconstruction claim, even when a candidate passes the other cases.
Group labels alone cannot certify that the real input contains one sheet.

## Bounded adaptive extraction

The initial half-cell extraction fails the 95%/quarter-cell agreement check on
the shifted, separately labeled sheet control. Its local-hull trimming rejects
triangles outside the observed boundary, but at coarse extraction resolution
this also discards a strip of supported surface near that boundary. Neither
initial local-implicit variant passes all conditional checks. The bubble's
appearance alone was therefore not used to select it.

`adaptive_surface_display.py` addresses this discretization loss. For each
group, it measures original-sample to **same-group triangles** distance. If
less than 95% of the samples are within a quarter cell, it halves extraction
spacing and repeats the same field construction, support trimming and defect
removal, for at most two refinements (0.5, 0.25, 0.125 cell spacing). Normal,
kernel, support and accuracy thresholds are unchanged. It does not enlarge a
group or change tracking to improve a score. Groups failing the bounded target
remain flagged; the three-million-grid-node resource limit still applies.

This correction was additionally checked on translated and rotated small
planes outside the original control table. Its acceptance criteria remain
exploratory and do not establish correctness for arbitrary geometry.

Results: `outputs/three_bubble_surface_display_adaptive_n064/`. This candidate
passes the conditional nonambiguous controls and real-event checks. The
same-group two-sheet counterexample still yields one false intermediate
surface, despite excellent sample distances. It is explicitly retained as a
failure of universal reconstruction. Do not use good sample fit as evidence
that the supplied group contains one physical shock sheet.

For the real event, only the two 177-sample groups require one refinement;
the outer group and the 226-sample group retain half-cell extraction. All four
groups meet the per-group target. The final display has five connected pieces,
91,440 triangles, no detected self-intersections/nonmanifold vertices/edges or
inconsistent winding, and 99.72% of samples within a quarter cell. Five pieces
for four supplied groups means the mesh is not forced to be completely closed
or to have one piece per group. Physical topology remains unvalidated.

## Artifacts and reproduction

Research outputs: `outputs/three_bubble_surface_method_research_n064/`.
Defect-checked outputs: `outputs/three_bubble_surface_display_checked_n064/`.
Adaptive candidate: `outputs/three_bubble_surface_display_adaptive_n064/`.
Their READMEs and CSVs report actual results and limitations. Existing production
defaults remain unchanged.

Use an isolated Python environment with NumPy, SciPy, Matplotlib, Pillow,
Open3D 0.19.0 and scikit-image 0.25.2. Plotly is optional for the interactive
comparison export. This session installed optional packages under
`/tmp/thesis-surface-research` and `/tmp/thesis-surface-support`; the project
and GPU/JAX environments were not modified.

```sh
python -m experiments.wind_bubble.research_surface_display \
  --output-dir outputs/three_bubble_surface_method_research_n064

python -m experiments.wind_bubble.finalize_surface_display \
  --research outputs/three_bubble_surface_method_research_n064 \
  --output-dir outputs/three_bubble_surface_display_checked_n064

python -m experiments.wind_bubble.adaptive_surface_display \
  --checked outputs/three_bubble_surface_display_checked_n064 \
  --output-dir outputs/three_bubble_surface_display_adaptive_n064

python -m experiments.wind_bubble.export_surface_display_viewer \
  --directory outputs/three_bubble_surface_display_checked_n064 \
  --adaptive outputs/three_bubble_surface_display_adaptive_n064 \
  --output outputs/three_bubble_surface_display_adaptive_n064/interactive_comparison.html

python -m unittest tests.experiments.test_implicit_surface_display \
  tests.experiments.test_display_mesh_cleanup \
  tests.experiments.test_adaptive_surface_display \
  tests.experiments.test_ball_pivot_mesh tests.experiments.test_surface_mesh
```

Use fresh output paths. These are CPU postprocessing commands; no new GPU
simulation or retracking is required. This is one static-event assessment;
temporal flicker, broad resolution behavior and arbitrary turbulent geometry
are not validated by it.
