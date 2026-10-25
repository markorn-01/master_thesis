# Callback timing correction

## Result

The independent thesis copy now schedules callback-only analysis at the requested
simulation times. Changes are in `astronomix/time_stepping/time_integration.py`.
The Riemann solver, hydrodynamic update formulas and shock-finder physics are
unchanged. The historical failure observations remain in
`../callback_timing_audit/`.

Before correction, regular callback targets 0, 0.125, 0.25, 0.375, 0.5 could
produce only 0 and 0.5 when one solver step crossed the intermediate times.
After correction, all five times are reached. An explicit early target at 0.05
also advances correctly instead of leaving the clock stalled there.

## Defined callback-only behavior

- Regular scheduling uses `num_snapshots` intervals over `[t_start, t_end]`,
  with an initial callback and one final callback: N+1 calls for N intervals.
- Explicit times are finite, strictly increasing and within the simulation
  interval. The initial time need not be listed. The final state is emitted
  once whether or not the final time is listed.
- Adaptive timesteps are shortened to land on the next analysis event and the
  final time. This can change the numerical trajectory relative to a run with
  no diagnostics. Equivalence comparisons must use matching event schedules.
- Fixed-step runs preserve the fixed timestep grid; requested events must align
  with it. Incompatible schedules raise an error rather than silently shorten
  steps and fail to reach the final time within the fixed iteration count.
- Schedule values must be concrete when calling the public wrapper, so they can
  be validated before entering JIT. Differentiating or externally JIT-compiling
  this callback wrapper with traced scheduling parameters is not supported.

## Compatibility

The stored-snapshot path is retained. Its regular output convention remains N
stored states with the final buffer slot reserved for the final state. If both
stored snapshots and callbacks are enabled, stored snapshots still take
precedence. The new callback schedule does not imply a change to that legacy
snapshot convention.

Callbacks still receive padded states. An online analyzer adapter will remove
ghost cells before analysis. The separate TO_DISK checkpoint driver is not
changed, and callback support is not added to it.

## Validation

Fourteen focused CPU tests passed with Python 3.10.11 and JAX 0.6.2, covering:

- regular events under large and smaller adaptive steps;
- explicit early and late times, omitted initial/final entries;
- nonzero start times;
- aligned fixed-step schedules and rejection of incompatible schedules;
- malformed explicit schedules;
- preserved stored-snapshot precedence;
- exact equality of evolving nonuniform gas states between callback and
  offline stored-snapshot runs at the same explicit times, including the final
  state. The state demonstrably evolves, so this is not a constant-state check.

The existing hydrodynamic regression suite also passed all **72 tests** in
114.201 seconds. Together with the 14 focused timing tests, 86 local tests
passed. Remote GPU validation of this correction remains outstanding.

The machine-readable timing report records eight callback-only schedules, all
matching their targets within float32 tolerance, plus one combined-mode probe
confirming the unchanged stored-snapshot precedence. `unpadded=False` is expected
and documents the retained callback API; it is not a failed timing check.

This is scheduling and state-delivery validation, not online shock-finder
validation, performance benchmarking, or GPU validation of these new changes.

## Reproduction

After syncing the corrected code, run from the repository root:

```bash
python -m unittest -v tests.time_stepping.test_callback_timing
```

To regenerate timing observations separately from the committed local output:

```bash
python -m experiments.shock.check_callback_timing --output-dir outputs/callback_timing_validation_iwr
```

No commit or push was made as part of this change.
