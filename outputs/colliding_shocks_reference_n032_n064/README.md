# Known-reference collision benchmark

Completed locally with JAX 0.6.2 on CPU. No simulation was evolved and no
production finder code was changed. The 96 cases cover 32^3/64^3 planar inputs,
incoming/outgoing fronts, several separations, exact conservative cell averages,
synthetic broadening, isolated-shock controls, contacts, and the meeting instant.

- `verification.json`: physical reference values, flux residuals, control
  outcomes, scope, and source hashes. Completion does not imply all difficult
  cases passed; the CSV files explicitly record the failures.
- `summary.csv`: one row per input case.
- `front_metrics.csv`: location coverage and Mach error for each known front.
- `profiles/`: representative centerlines of the analyzed 3D fields, including
  the production detection masks. These are not full 3D snapshots.
- `recovery_vs_separation.png`: location coverage and Mach error separately.
- `known_front_profiles.png`: pressure profiles and known/detected positions.

Isolated and widely separated exact shocks pass. A two-cell incoming pair
returns M = 1 instead of 3; a two-cell outgoing pair produces no selected
surface despite four centerline zone candidates. See the separate
`../colliding_shocks_sampling_trace_n064/` artifacts and
`../../experiments/shock/COLLIDING_SHOCKS.md` for causes and limitations.
