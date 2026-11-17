# Reconstruction research: raw candidates

This directory contains 25 controlled reconstructions (five cases × five
methods) and five reconstructions of the existing 64³ three-source event at
t = 0.16. No simulation or tracking was rerun.

Read `experiments/wind_bubble/SURFACE_DISPLAY_RESEARCH.md` for the literature,
equations, frozen parameters, reproduction commands and interpretation limits.
The display candidates are guarded single-radius ball pivoting, compact local
implicit MLS with saved or display-fit normals, support-trimmed Poisson, and
untrimmed Poisson as an extrapolation diagnostic. Raw Poisson is not included
as a recommended display in the comparison figure.

The algorithms do not all preserve mesh vertex locations. Ball pivoting uses
original samples. MLS and Poisson create separate approximate display vertices;
their NPZs retain original positions, physical normals, Mach values and labels
under explicit `original_*` keys. The physical analysis products are unchanged.
Nearest-sample attribution does not make a new display vertex a measured state.

`summary.csv` contains geometric measurements, including point-to-triangle
distances, rather than only nearest-mesh-vertex distances. Native Open3D
self-intersection and vertex-manifold checks add tests not present in the
earlier ball-pivoting comparison. `verification.json` passing means the
construction/readback checks completed; it does **not** mean these raw meshes
all passed geometry tests or recovered the physical surface.

At the real event, before conservative defect removal:

| Method | Edge-connected pieces | Samples within 0.25 cell of triangles | Nonmanifold vertices |
| --- | ---: | ---: | ---: |
| Guarded ball pivoting | 299 | 92.44% | 770 |
| Local implicit, saved normals | 8 | 99.57% | 52 |
| Local implicit, display-fit normals | 6 | 99.62% | 44 |
| Poisson with support trimming | 24 | 99.43% | 38 |

These raw counts motivate defect removal and a new readback. More triangles,
fewer pieces or small sample distances do not prove correct topology. The
deliberate same-group two-sheet counterexample exposes invented intermediate
surfaces in all families, including apparently smooth outputs.

For the defect-checked figures, conditional eligibility assessment and
interactive comparison, use `outputs/three_bubble_surface_display_checked_n064/`.
Neither directory replaces production defaults or validates an entire animation.
