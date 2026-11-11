# One branching history: t=0.07 → 0.08

Open `branching_history.png`. On the left, the two early provisional
segments S000/S001 point to the later segment S002 through **candidate
predecessor** edges. On the right, each segment retains its own saved Mach
history. No interpolated line joins the different measured populations.

At t=0.07 each earlier group contains 5,064 samples. At t=0.08 the single
retained group contains 11,424 samples. At both association distances,
each parent supports a distinct set of 5,712 child samples: half the child
group, with no overlap between these two support sets. Together they cover
the child group. Counts at successive events are not conserved particles,
mass or area; samples are redetected from each state.

`candidate_graph.json` records three segment nodes and two candidate edges.
`candidate_predecessor_evidence.csv` records the evidence separately at the
two tested distances (four rows). Existing segment IDs, history values and
matcher decisions remain unchanged. S002 does not inherit either physical
identity, and no confirmed physical shock merger is inferred.

The support sets and exclusive contributions are independently recomputed
using pointwise neighborhood queries. Boundary sample counts and Mach
medians are checked against the source catalogs. JSON readback and unchanged
input hashes are checked; source/input hashes appear in `verification.json`.
The figure was rendered and visually inspected. This is one reviewed
example, not an automatically validated general merger tree.

Reproduce with NumPy, SciPy and Matplotlib, in a new output directory:

```sh
python -m experiments.wind_bubble.plot_branching_history --output-dir outputs/two_bubbles_branching_history_n128
```

No GPU or new simulation is required. The early Mach histories overlap
because of the symmetric setup. The later median is computed from all
valid Mach samples in the later group; it is not an average of earlier
group medians. The shaded interval marks the two saved times bracketing
the grouping change, not an exactly determined physical merger time.
