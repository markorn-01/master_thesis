# Saved bubble joint-reconstruction transfer

**Result: restricted model does not transfer; no detector change.**

See the [protocol and interpretation](../../experiments/wind_bubble/BUBBLE_JOINT_RECONSTRUCTION.md).

- 40 native outer-shock windows: no accepted joint or shock-only reconstruction.
- 40 central candidates: eight admissible cells each, below the unchanged
  prototype's 16-cell minimum; not fitted across sources or the midplane.
- No new simulation, altered threshold, or physical shock-absence claim.

Files:

- `verification.json`: verdict, fixed settings, input/source fingerprints.
- `summary.csv`: all 80 window outcomes and historical nearby surface flags.
- `fits.jsonl`: all 40 fits, including rejected model parameters and restarts.
- `fitted_profiles.npz`: native five-variable windows keyed by case ID; columns
  are x samples and rows are density, x/y/z velocity, and pressure.
- `bubble_transfer.png`: saved outer profiles against rejected model predictions.

Rejected candidate parameters are not accepted physical Mach or position
measurements. This transfer failure does not undo the controlled planar
feasibility result, and it does not resolve the physical nature of the central
collision layer.
