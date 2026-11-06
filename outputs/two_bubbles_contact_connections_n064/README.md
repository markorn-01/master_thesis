# Inspecting connections near two-bubble collision

This is CPU postprocessing of saved 64³ GPU catalogs and gas slices. Neither the
simulation nor the shock finder was modified. Source positions are display
markers, not clustering features or ownership labels.

## What was inspected

The reference partition uses a proximity radius of 1.5 grid cells and signed
normal angles no larger than 30°. We inspect the edges that join different
**raw reference components** when the radius and/or angle are relaxed.
All calculations use 3D positions. The figures show a thin projected section
over the saved pressure/density plane; the plane is z=0.5078125, with displayed
points within ±1.5 grid cells. This is not an exactly co-located slice of every
displayed sample.

| Time | r=1.5, 30° | r=1.5, 60° | r=2.5, 30° | r=2.5, 60° |
| --- | ---: | ---: | ---: | ---: |
| .06 | 2 | 2 | 2 | 2 |
| .08 | 2 | 1 | 2 | 1 |
| .09 | 2 | 1 | 2 | 1 |
| .10 | 1 | 1 | 1 | 1 |

These are raw connected-component counts, not validated physical shock counts.

## Findings

- Before contact at .06, none of these settings joins the two components.
- At .08 and .09, changing the angle threshold alone from 30° to 60° is
  sufficient to join them. Increasing distance alone is insufficient.
- For r=2.5/60°, there are 60 joining edges at .08 and 540 at .09. At .08 their
  endpoint normal angles span 56.29–59.62° (median 57.82°); at .09 they span
  36.78–59.52° (median 49.20°). Their median lengths are 1.304 and 1.566 cells.
- The figures place these edges near the neck of the colliding bubbles. Their
  chord/normal alignment medians are about .236 and .141 (0 is tangential,
  1 is along the normal). This is compatible with connections along a bending
  surface, rather than sufficient evidence that every joining edge is spurious.
  Geometry at this resolution and the projected gas slices do not settle the
  physical interpretation independently.
- At .10 the stricter reference already has one component. Consequently zero
  *joining edges relative to that partition* does not mean a lack of connections.

## Files

- `event_0006_connections.png`, `event_0008_connections.png`,
  `event_0009_connections.png`, `event_0010_connections.png`: pressure, density,
  and normal panels. Point colors are reference components. Pink lines are
  relaxed-graph edges crossing reference components. All qualifying lines in
  the display slab are shown; no random edge subsampling.
- `event_*_joining_edges.csv`: original catalog row IDs, endpoint positions,
  component IDs, length in cells, angle, chord alignment, Mach and density jump.
  Mach and density jump are context only and are not clustering features.
  An edge appears in each parameter setting that accepts it; rows across
  settings should not be summed as independent edges.
- `summary.csv`: all four parameter combinations at each inspected time.
- `metadata.json`: input and analysis-source hashes.
- `verification.json`: input integrity, agreement with the earlier graph study,
  and saved edge geometry/accounting checks. These do not validate shock identities.
- `SHA256SUMS`: output checksums, excluding the manifest itself.

## Consequence for tracking

We must distinguish a connected shock surface from the historical origin of its
parts. After collision, one connected outer surface need not retain two separate
spatial cluster labels merely because there were two sources initially.
Persistent tracking should allow a possible two-to-one merge; source ancestry
would be additional information, not the same thing as surface connectivity.
This inspection motivates that distinction but does not implement or validate
such tracking. FuzzyCat and smoother mesh displays remain subsequent work.

Reproduce in an environment with NumPy, SciPy and Matplotlib:

```bash
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.inspect_contact_connections \
  outputs/two_bubbles_iwr_compgpu9_n064/online \
  --output-dir outputs/two_bubbles_contact_connections_n064_new
```

The output directory must not already exist. No GPU is required.
