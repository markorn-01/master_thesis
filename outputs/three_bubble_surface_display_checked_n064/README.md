# Defect-checked comparison before adaptive extraction

These 24 outputs apply the same conservative defect removal to four methods
on five controls and one saved three-source event. Every triangle incident to
a nonmanifold vertex or a detected intersecting pair is removed. No coordinate
is relocated, and no replacement face is introduced. The saved oriented
triangle coordinates are checked against the raw research outputs, and the
NPZ arrays are read back exactly.

The raw reconstruction inputs are in
`outputs/three_bubble_surface_method_research_n064/`.

`eligibility.json` shows that **none of these initial variants passes every
conditional display criterion**. Ball pivoting loses too much real-event
coverage after defect removal. Both local-implicit versions lose too much
boundary coverage on the small shifted-sheet control. Poisson fails the
planar-hole coverage and separate-sheet displacement checks. These failures
were retained, not waived based on the appealing bubble images.

This motivated the bounded extraction-resolution correction in
`outputs/three_bubble_surface_display_adaptive_n064/`, which is the main
candidate for review. Its README explains its conditional success and the
remaining same-group sheet ambiguity.

`verification.json` verifies the readback/provenance and recorded geometry
measurements; passing does not mean all methods satisfy eligibility.
`cleanup.json` records removals. `summary.csv` contains the measurements.
The PNGs show full and cutaway views with unchanged group labels. Mesh pieces
are not physical shock counts. The field `cross_nearest_truth_sheet_faces`
is only a nearest-input-point truth-label proxy for generated vertices; it
can report mixed nearest labels even on a correctly positioned plane. Actual
known-plane offset and supplied-group membership are the relevant sheet checks.

No production default, physical diagnostic or tracking decision is changed.
