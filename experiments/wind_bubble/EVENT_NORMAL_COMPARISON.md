# Full-event display-normal comparison

Compared all 25,136 saved surface samples at t=0.16 on the 128³ two-bubble
run. Original refined positions, Mach values and event-local grouping are
identical in all three modes. Reconstructed with single-radius ball pivoting
(radius one grid spacing), followed by the same original geometric guards.
Unlike the earlier patch comparison, this reruns ball pivoting itself.

Covariance-plane fits use same-group neighbors within either two or three
grid spacings, now on all three retained groups. Fit signs follow the saved
physical normals. All fitted points have numerically valid plane fits here;
this does not guarantee correct surface membership or resolved physics.

## Whole-event results

| Measurement | Original normals | Fit: 2 cells | Fit: 3 cells |
| --- | ---: | ---: | ---: |
| Raw proposed triangles | 49,949 | 49,968 | 49,968 |
| Accepted triangles | 35,408 | 36,497 | 36,641 |
| Samples used by accepted triangles | 22,915 | 23,389 | 23,460 |
| Unused samples | 2,221 | 1,747 | 1,676 |
| Edge-connected mesh pieces | 830 | 688 | 749 |
| Open boundary edges | 17,758 | 18,053 | 17,977 |
| Nonmanifold edges | 0 | 0 | 0 |
| Inconsistent shared-edge winding | 0 | 0 | 0 |

Mesh pieces are connected through shared edges, not merely shared vertices.
They are not physical shocks or persistent identities. Open mesh edges do
not establish physical holes. Their count can increase as more samples and
triangles become represented, so improved vertex coverage does not imply
fewer gaps or a closed surface.

## Inner groups

| Group and measurement | Original normals | Fit: 2 cells | Fit: 3 cells |
| --- | ---: | ---: | ---: |
| Group 1: accepted triangles | 1,812 | 2,104 | 2,080 |
| Group 1: connected pieces | 155 | 88 | 108 |
| Group 1: largest piece, faces | 236 | 1,307 | 1,191 |
| Group 1: open edges | 1,366 | 1,252 | 1,236 |
| Group 2: accepted triangles | 1,804 | 2,095 | 2,071 |
| Group 2: connected pieces | 157 | 90 | 109 |
| Group 2: largest piece, faces | 234 | 1,264 | 1,147 |
| Group 2: open edges | 1,398 | 1,285 | 1,269 |

The outer group has 518, 510 and 532 pieces, respectively. Its largest piece
increases from 12,738 to 28,179 and 28,307 faces. The whole-event reduction
in piece count therefore does not mean every group or gap improves.

The two-cell fit has fewer pieces and greater inner-group vertex coverage
than the three-cell fit; the three-cell fit has slightly better total vertex
coverage and fewer total open edges. No single setting dominates all metrics.
The two-cell fit is a reasonable candidate for a limited temporal display
check, rather than an adopted production default.

## Validation and limits

The original-normal reconstruction exactly reproduces the saved baseline
face array. Each output is reopened and its positions, saved physical
normals, Mach values and labels checked exactly against the input. Every
accepted face passes the independent geometric checker with its actual
display-normal set. Areas are cross-checked with Heron's formula as a
numerical check only, not substituted into any physical area diagnostic.
Brute-force neighborhood counts and independent SVD directions check 100
deterministically spaced points per group and radius: 600 valid fits.
The 13 focused plane-fit, patch-angle and temporal-geometry tests pass.

All three cross-group distance/maximum-edge separation bounds pass in each
mode, excluding intersections between the different labeled groups for
this event. This is a sufficient bound using unchanged group geometry,
not proof against within-group bridges or self-intersections. The known
close-sheet failure when both sheets share one group label remains.

No shock finder, physical normal, Mach estimate, density jump, heating
measurement or temporal association was changed. Point positions are not
smoothed. Fitted normals affect only this experimental display reconstruction.
One event cannot establish temporal stability or smoothness of an animation.

## Outputs and reproduction

`outputs/two_bubbles_event_normals_n128/` contains three explicitly separated
mesh NPZs, aggregate and per-group CSVs, a checked-readback report with source
and input hashes, and two figures:

- `event_normal_comparison.png`: full meshes, display cutaways, inner groups;
  fixed camera, spatial bounds per row, and Mach scale.
- `central_sections.png`: actual triangle intersections with z=0.5; line
  gaps are display-mesh gaps, not validated physical openings.

```sh
python -m experiments.wind_bubble.compare_event_normals --output-dir outputs/two_bubbles_event_normals_n128
python -m unittest tests.experiments.test_patch_normals tests.experiments.test_mesh_patch tests.experiments.test_temporal_mesh
```

Requires the optional Open3D environment. The output directory must not
already exist. These are CPU postprocessing commands; no new GPU run is
needed. Next, compare the original and two-cell candidate at the same
three saved times used previously, retaining the three-cell setting as
sensitivity evidence rather than silently replacing the existing renderer.
