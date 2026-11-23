# Collision-layer physical check and one matched resolution experiment

The verified production audit explains the detection gap. A separate saved-state
analysis tests whether the dense ridge can independently be identified as a
resolved shock. Its evidence and limits are documented in
`outputs/three_bubble_collision_physics_n064/README.md`.

## Existing-state analysis

```bash
python -m experiments.wind_bubble.assess_collision_physics \
  --root outputs/three_bubble_collision_detection_audit_iwr_compgpu8_n064 \
  --output-dir outputs/three_bubble_collision_physics_n064
```

The completed local result is already in that output directory; use a fresh
directory to repeat it. This is a read-only NumPy/SciPy analysis. It changes no
production detector and requires no new hydrodynamic run.

It tests pressure-oriented axial state pairs and positive-pressure-gradient
rays, independently of temperature normals. It records entropy, the thermal
Hugoniot relation, momentum, total enthalpy, tangential velocity, and necessary
shock-frame sonic ordering. Candidate speed is inferred from the mass jump,
not measured independently. All sampled pairs are retained, including failures.

At 64^3, the later central contrasts fail single-shock consistency. However,
the coarse outer forward-front comparison also often lacks suitable clean
endpoint states, so lack of a compatible pair cannot prove physical absence.

## Matched 128^3 experiment

`run_collision_resolution_check.py` reuses the frozen production integrator,
three-source runner and audit functions. The reference is the verified 64^3
GPU audit, including its captured-state and source fingerprints.

Kept fixed: three equal sources and their positions, gamma=5/3, ambient rho=1
and P=0.01, physical injection radius 0.0625, wind speed 10, all 17 original
float32 event times, end time 0.16, and the original maximum timestep cap.
The grid changes to 128^3. Adaptive timesteps may differ because the grid is
finer; complete states are not expected to match across resolutions.

Each source retains prescribed mass rate 0.006565141402540684 and mechanical
energy rate 0.3282570701270342 in code units. The 128^3 injection mask has 1736
cells per source. The base mass parameter adjusts from 0.011 to approximately
0.008110599078341014 so that the integrated discrete source coefficients stay
fixed. These are prescribed budgets, not measurements of net energy deposited
into the evolving gas.

The observer captures events 4, 6, 8, 12 and 16. Five full states require about
200 MiB; this is deliberate validation output. No full offline second run or
temporal clustering is added. The runner first verifies configuration/budget,
schedule and production mask/export invariants, then automatically writes
the independent physical assessment to the new run's `physics/` directory.

On the confirmed host compgpu8, GPU 6, after pushing and pulling the code:

```bash
export CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_PREALLOCATE=false

"$HOME/.venvs/master-thesis-py312/bin/python" \
  -m experiments.wind_bubble.check_gpu_environment

"$HOME/.venvs/master-thesis-py312/bin/python" -m unittest \
  tests.experiments.test_collision_physics \
  tests.experiments.test_collision_resolution_check \
  tests.experiments.test_collision_detection_audit \
  tests.experiments.test_detection_audit

"$HOME/.venvs/master-thesis-py312/bin/python" \
  -m experiments.wind_bubble.run_collision_resolution_check \
  --require-gpu \
  --output-dir outputs/three_bubble_collision_resolution_iwr_compgpu8_n128
```

Use a fresh output directory. If a pull conflicts with the existing 64^3
audit outputs, stash that specific directory before pulling; do not remove it.

### Recover the completed simulation after a postprocessing error

The first 128^3 run reached t=0.16 and saved its five states, but the wrapper
passed an argument namespace instead of a directory path to the audit function.
That call is corrected. The integration does not need to be repeated:

```bash
"$HOME/.venvs/master-thesis-py312/bin/python" \
  -m experiments.wind_bubble.run_collision_resolution_check \
  --audit-only --require-gpu \
  --output-dir outputs/three_bubble_collision_resolution_iwr_compgpu8_n128
```

Use the SAME directory as the completed run. This mode checks completion,
resolution, configuration, schedules, saved-state shapes/finiteness and all
recorded production source hashes before auditing. Only this wrapper's own
hash may differ to repair orchestration; both old and new hashes are recorded.
It does not invoke the time integrator or change the saved simulation states.
Saved state and primary-output hashes are checked again after the audit.
The result records `resumed_from_saved_states: true`.

A regression test exercises the recovery dispatch, requires the auditor's
first argument to be a Path, and verifies that integration is never called.

## Interpretation and stopping condition

Compare the same physical cuts, the independent consistency checks, and the
production detection masks at both resolutions. Fixed-cell and fixed-physical
sampling widths are included. A `passed: true` resolution audit verifies setup
and instrumentation; it does not assert shock recovery or physical convergence.

One successful state-pair fit does not establish a unique physical shock,
and failure to fit does not prove absence. Look for separated fronts and
consistent states across sampling choices, alongside the outer-front check.
If the finer run still cannot separate them, retain a resolution-limited
conclusion rather than endlessly changing detector thresholds.

## Completed comparison

The matched GPU run and saved-state recovery completed successfully. The local
review verified the copied hashes, compared a common physical axis, and located
all 124 post-contact pairs passing the 20% consistency level. All 54 sampled
pressure jumps reaching the Mach-1.3 reference are covered by an existing
exported grid cell on their axial segment or at their ray center. The central
ridge is not established as a missed isolated shock in these two runs.

The completed report, figures, qualifications and suggested explanation are in
`outputs/three_bubble_collision_resolution_review_n064_n128/README.md`.
Reproduce the read-only review with a fresh output directory:

```bash
python -m experiments.wind_bubble.review_collision_resolution \
  --output-dir outputs/three_bubble_collision_resolution_review_n064_n128
```

This closes the bounded saved-data/resolution check. It neither proves shock
absence nor certifies convergence or complete physical shock recovery.
