# Callback timing audit — historical pre-correction findings

This records behavior **before** the callback correction. Current results are
in `../callback_timing_validation/`. The integrator was subsequently changed
only in the independent thesis repository, following the user’s clarification.

## Scope and status

Six small 1D uniform hydrodynamic runs exercised the public `time_integration`
API. The integrator, solver and shock finder were not modified. The added audit
script is `experiments/shock/check_callback_timing.py`. Results were obtained
locally with Python 3.10.11 and JAX 0.6.2 on CPU; this audit has not been run on
IWR. It reports readiness checks, not a passing regression-suite result.

The original plan requested coordination for solver-internal changes. The user
subsequently clarified that this is an independent copy of the implementation;
changes here do not modify Niklas’s code.

## Reproduced timing behavior

| Probe | Intended analysis times | Observed callback times |
| --- | --- | --- |
| Regular, large adaptive step | 0, 0.125, 0.25, 0.375, 0.5 | 0, 0.5 |
| Regular, smaller adaptive steps | 0, 0.125, 0.25, 0.375, 0.5 | 0, 0.193649, 0.387298, 0.5 |
| Explicit later times | 0, 0.20, 0.35, 0.50 | All matched within float32 tolerance |
| Explicit early time, bounded fixed-step probe | 0, 0.05, 0.31, 0.50 | 0, 0.05; final callback was still at 0.05 |
| Stored snapshots plus callback | 0, 0.20, 0.35, 0.50 | No callbacks; stored snapshot times matched |
| Regular, nonzero start | 0.20, 0.325, 0.45, 0.575, 0.70 | 0.20, 0.70 |

The regular schedule is interpreted here as `num_snapshots=4` intervals plus
one final callback, consistent with the callback predicate and final recorder.
This is not the stored-snapshot buffer convention, which has four total slots
and reserves the last for the final state. Choosing a documented convention is
part of the proposed correction.

All hydrodynamic outputs remained finite and uniform density was preserved.
Every observed callback received shape `(3, 20)` including ghost cells; the
physical input and returned final state had shape `(3, 16)`.

## Explanation

- The regular callback predicate checks whether a target has been passed, but
  the adaptive step is not clamped to that target. One solver step can pass
  several analysis times; missed states cannot be recovered afterward.
- The explicit-time timestep clamp and callback predicate use different
  schedules. At t=0.05 the next explicit target is reached, but the regular
  predicate still waits for t=0.125. The callback index does not advance and
  subsequent steps are clamped to zero. A fixed iteration count bounded this
  reproduction. An adaptive early-time run was deliberately not attempted;
  a nonterminating adaptive loop is a consequence suggested by the code, not
  an independently executed result.
- Stored snapshots take precedence through an `if/elif` branch, bypassing the
  callback when both flags are enabled.
- Padded callback input is the existing API behavior, not itself a timing bug.
  The future online adapter can unpad it without changing the callback API.

## Proposed minimal correction for review

1. Make explicit-time callback readiness use the requested explicit timepoints,
   consistently with the timestep clamp. Guard schedule exhaustion and ensure
   the integration still reaches the final time.
2. Define regular callback counting and endpoint behavior; clamp adaptive steps
   to the next requested event when exact regular analysis is requested.
3. Preserve the default stored-snapshot workflow and its validated outputs.
4. Keep padded input compatibility and unpad inside the online-analysis adapter.
   Combined stored-snapshot/callback support can be a separate decision; it is
   not required for the first callback-only online runner.
5. Add regression tests for early and late explicit times, initial/final calls,
   nonzero start time, event counts, and progress to the requested final time.
   Test solver-state equivalence on matching schedules before production use.

No changes to the Riemann solver, shock-finder physics, Mach estimator, or
thermalization formula are proposed. No runtime or memory improvement is
claimed yet.

## Reproduction

From the repository root in a compatible environment:

```bash
python -m experiments.shock.check_callback_timing
```

To keep IWR results separate, use a new output directory with `--output-dir`.
`callback_timing.json` contains the full precision observations and environment.
