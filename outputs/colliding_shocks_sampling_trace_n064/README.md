# Two close-front failure traces

Production code is unchanged. Inputs come from the completed 64^3 exact-state
benchmark in `../colliding_shocks_reference_n032_n064/`.

`sampling_trace.csv` and `sampling_crosses_opposite_shock.png` compare incoming
Mach-3 fronts separated by eight and two cells. At two cells, the nominal
upstream endpoint lies beyond the opposite front: P_pre = P_post = 11 gives
reported M = 1. The dimension-independent production sampler reproduces the
saved 3D Mach values.

`surface_comparisons.csv` and `surface_selection_opposing_tie.png` show an
outgoing pair with known Mach 1.914854 and two-cell separation. All four
centerline zone candidates share the same divergence minimum and fail the
backward comparison. The inner two compare against opposite normals and
reject each other. The reconstructed comparisons reproduce the saved 3D mask.

`verification.json` records checks and source/input hashes. These are specific
planar sampling and surface-selection failures; neither trace alone explains
the bubble case's earlier rejection by the thermodynamic zone criteria.
