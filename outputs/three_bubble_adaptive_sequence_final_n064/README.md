# Checked three-bubble display sequence after final cleanup

This sequence corrects the combined-mesh intersection at event 5, simulation
t approximately 0.05. Independent group refinement introduced 34 intersecting
pairs involving 35 faces. A final combined cleanup removes those 35 faces
(0.031% of the candidate) without adding triangles or moving retained vertices.
The small resulting gap is retained.

All sixteen populated events now pass the existing geometry and own-group
sample-agreement checks. Event 0 remains empty. At event 5, all three groups
retain 100% sample agreement within 0.25 grid cell. The other sixteen event
mesh files are byte-identical to the parent sequence. Physical sample arrays,
group labels, tracking segments and fitting/support thresholds are unchanged.

- `repair_0005.png`: local before/removed-faces/after comparison.
- `repair_0005.json`: intersection audit, cleanup accounting and group checks.
- `summary.csv` and `group_accuracy.csv`: final event and group measurements.
- `verification.json`: provenance hashes and regression checks.
- `../three_bubble_adaptive_animation_final_n064/`: corrected full animation.

The original rejected sequence remains in `../three_bubble_adaptive_sequence_n064/`.
Reproduction commands and limitations are in
`../../experiments/wind_bubble/FINAL_COMBINED_MESH_CLEANUP.md`.
These are display meshes; passing the checks does not establish physical
topology, resolve uncertain ancestry or solve the same-group close-sheet
counterexample. Unsupported holes remain visible.
