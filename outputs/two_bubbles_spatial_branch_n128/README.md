# Spatial view of the reviewed branching relationship

Open `spatial_branch.png`:

- Left: original refined detected positions at t=0.07; orange S000 and
  green S001, each with 5,064 samples.
- Center: the two previously checked candidate predecessor arrows into
  S002. Graph layout is schematic, not a particle trajectory.
- Right: original positions at t=0.08; blue S002 contains 11,424 samples.

Every point in each retained group is displayed without subsampling,
positional smoothing or triangles. Both spatial panels use the same
camera and bounds. Colors identify provisional history segments, not
Mach values, gas density or established physical shock identities. Dense
points overlap visually; the apparently filled shapes are point plots.

The grouping changes from two groups to one. Both arrows are supported
at both tested distances, as recorded in the existing branching-history
evidence. This figure adds no clustering, matching decision or physical
merger claim. It is a spatial explanation of one reviewed relationship.

Displayed group IDs are taken from the saved history table and mapped to
the original catalog partitions. Counts, complete point selection, input
hashes and image decoding are checked. Coordinate hashes and all original
inputs appear in `verification.json`. The rendered figure was visually
inspected. Existing history and branch files remain unchanged.

Reproduce using NumPy and Matplotlib, in a new output directory:

```sh
python -m experiments.wind_bubble.plot_spatial_branch --output-dir outputs/two_bubbles_spatial_branch_n128
```

No GPU, new simulation, reconstruction or general tracking algorithm is
required. This completes the spatial presentation of S000/S001 → S002.
