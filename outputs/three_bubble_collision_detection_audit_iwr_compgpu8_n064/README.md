# Verified GPU audit of the lower two bubbles' interaction

The matching three-bubble replay ran on compgpu8 GPU 6, at 64^3 cells with
the original 17-event schedule ending at t=0.16. Detector physics, thresholds,
source prescription and timestep cap were unchanged. Grouping and meshing
were not used in the audit.

## Verification and local copy

`audit/verification.json` reports a passing comparison of all 17 events'
surface catalogs, histograms and gas slices against the original GPU run.
Production mask decomposition and reconstructed export indices/positions
also passed. The local copy was checked against all five captured-state
hashes, all recorded input hashes and source hashes; every check passed.
The captured arrays are finite and have shape (5, 64, 64, 64).

The reference comparison was repeated successfully on the copied files.
`compared_captured_states: 0` is expected: the original reference retained
no full gas snapshots. This is not complete full-state online/offline
equivalence and does not independently validate the physical shock model.

## Established rejection mechanism

The central diagnostic box is x=0.45–0.55, y=0.335–0.415, z=0.42–0.58.
It contains 360 grid centers; its saved central XY plane at z=0.5078125
contains 36. Counts apply to this explicit region, not every possible
interacting front.

Central-slice counts:

| Time | Cells | Compression | Alignment | Pressure jump | Temperature jump | Zone | Surface |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.04 | 36 | 32 | 20 | 32 | 36 | 20 | 16 |
| 0.06 | 36 | 28 | 0 | 0 | 36 | 0 | 0 |
| 0.08 | 36 | 12 | 0 | 0 | 36 | 0 | 0 |
| 0.12 | 36 | 8 | 0 | 0 | 36 | 0 | 0 |
| 0.16 | 36 | 28 | 0 | 0 | 36 | 0 | 0 |

At t=0.08, 0.12 and 0.16, all 360 cells in the 3D box also fail alignment
and the pressure-jump test. The zone is empty before surface selection.
No selected surfaces in this box are removed by source exclusion/export.
This establishes that the checked late gap originates in shock-zone
detection, before grouping, tracking or display reconstruction.

At t=0.06 the slice is already empty, but there are still 56 selected grid
centers in the 3D box, of which 48 refined positions remain inside the box.
Thus earlier thin-slice absence must not be confused with absence at every
nearby depth.

## Why a pressure peak can still fail the jump test

The resolved profiles show density and pressure rising toward the central
layer while the temperature proxy T=P/rho falls. The full 3D temperature and
density gradients have a negative dot product in the later checked box,
failing the required positive alignment.

The jump sampler follows the normal -grad(T)/|grad(T)|, assigning the hotter
side as the nominal post-shock side. Across this layer the pressure change
has the opposite sign under that assignment. A visible pressure peak is
therefore not sufficient to pass the directed pressure-jump criterion.

For example, at t=0.08 and (x,y,z)=(0.4609375,0.3671875,0.5078125):

- Velocity divergence is approximately -9.218, so compression passes.
- grad(T) dot grad(rho) is approximately -7139, so alignment fails.
- The sampled nominal P2/P1 is approximately 0.686, below the required 1.8625.
- The sampled nominal T2/T1 is approximately 6.759, above the required 1.29218.

These are local detection-stage samples. Their nominal side labels are not
an independent identification of the physical upstream/downstream states.
In particular, this is not merely a pressure increase that narrowly misses
the Mach threshold: this example has a negative directed log pressure jump.

## Figures to show Tobias

1. [Cross-layer profiles at t=0.08](audit/collision_profiles_0008.png):
   density/pressure rise, temperature falls, and the alignment dot product
   is negative. Velocities are in the simulation frame; opposing x velocities
   do not alone establish negative full 3D divergence everywhere.
2. [Production masks at t=0.08](audit/production_masks_0008.png): teal means
   passing/selected, gray means failing. The black box is the counted slice
   region. Alignment and pressure jump fail there, leaving no zone/surface.

Both figures were visually inspected after copying from the GPU host.
`audit/collision_stage_counts.csv` and `audit/cross_collision_profiles.csv`
provide the underlying counts and resolved profiles at every captured time.

## Scope of the answer

The audit identifies why the present finder produces the gap. It does not
prove that the dense layer contains no physical shock, classify it as a pure
contact, or establish that the criteria should be weakened. Recovering any
overlapping or underresolved physical fronts is a separate accuracy question.

Suggested explanation:

> I replayed the same setup and checked the original detector stages. The
> gap appears before grouping or surface smoothing. In the central dense
> layer, density and pressure rise while temperature falls, so the alignment
> test fails. The pressure-jump test along the temperature-based sampling
> direction also fails. At t=0.08 neither test passes in any of the 36 checked
> slice cells, so no shock-zone candidates reach surface selection. This
> explains the missing detections, while the physical interpretation of any
> overlapping or unresolved fronts remains a separate question.
