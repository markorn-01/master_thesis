# Evolved planar collision: positive reflected-front control

See [setup, results, early failures, and limits](../../experiments/shock/EVOLVED_COLLISION.md).

Six genuine Astronomix 1D hydrodynamic runs: incident Mach 3 and 10, with
256/512/1024 cells and 81 saved times each. The production 3D finder analyzes
unchanged thin extrusions of evolved profiles.

All 276 individual front checks in the declared separated regime pass the
two-cell position and 5% Mach targets. Independent numerical-flow controls pass.
Close-front failures on the coarsest grid remain in the tables. This does not
model late fading in hot bubble cavities or resolve the original bubble case.

- `verification.json`: gates, outcomes, solver metadata, and source fingerprints.
- `events.csv`: all events with fluid errors and the separated-regime flag.
- `fronts.csv`: known targets and detection outcomes, including misses.
- `M*_n*.npz`: evolved primitive profiles `(event,x,rho/u/P)`, actual times,
  detection fields, and reference trajectories used only for scoring/overlays.
- `evolved_collision_spacetime.png`: density/pressure/velocity through time,
  exact trajectories, and detected surface positions.
- `evolved_collision_profiles.png`: incoming, close reflected, and separated
  reflected profiles versus exact cell averages.

No physical shock is prescribed at the post-collision midplane. Full profiles
retain numerical center artifacts. The unrelated native 1D last-zone loop fix
is documented separately from the unchanged 3D analysis path.
