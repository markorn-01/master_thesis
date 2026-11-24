# Joint reconstruction feasibility checkpoint

This is a CPU experiment on controlled planar profiles, not a bubble rerun.
See [method, command, safeguards, and limitations](../../experiments/shock/JOINT_SHOCK_CONTACT_FIT.md).

- `verification.json`: fixed settings, decision, source/input fingerprints.
- `summary.csv`: all 19 case outcomes, including negative and mismatch controls.
- `profiles/`: the density/velocity/pressure observations given to the fitter.
- Individual JSON files: scoring truth kept separate from the fit, all tested
  blur/orientation hypotheses, restarts, residuals, and inferred parameters.
- `parameter_scans.csv`: main-case fixed-Mach and fixed-gap refits.
- `joint_reconstruction.png`: hidden-state example, sensitivity scan, and limits.

The 10 positive cases support a joint reconstruction under the declared fan
and blur assumptions. The nine other cases receive no usable joint claim.
The Gaussian-blurred input is deliberately a *known shock* outside the fitted
blur family: its rejection is a model limitation, not a shock-absence finding.
Best joint-candidate parameters remain in JSON even for rejected models and
must not be treated as accepted physical measurements.

Production physics, detection thresholds, bubble outputs, and simulations were
not changed. This checkpoint does not resolve the actual bubble collision layer.
