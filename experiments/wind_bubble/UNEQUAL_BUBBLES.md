# Prepared unequal-wind validation (remote collision run pending)

## Fixed experiment

Start at 64³ with the same two source locations, injection radius 0.0625,
ambient state, wind speed 10, 17 events and end time 0.16 as the saved equal
case. Set mass/luminosity multipliers to **1 and 0.5**. The source at x=0.35
retains the reference rate; x=0.65 receives half its mass and prescribed
mechanical-energy coefficient. Total injected coefficients are consequently
75% of the equal two-source setup. This is deliberate asymmetry, not a
fixed-total-energy comparison. No prediction of exact collision time is used.

Optional `WindConfig.source_strengths` supports explicit nonoverlapping 3D FV
EI sources only. Empty weights retain identical-source behavior. Weighted
energy rates are applied before the existing state-dependent pressure
conversion. Source-aware timestep estimation uses the same injection function,
so it sees the weighted hypothetical state automatically. No integrator or
shock-finder estimator is changed.

Metadata retains the shared base wind parameter for compatibility and adds
the explicit weights, actual per-source mass parameters and luminosities.
`source_budget_plan` documents the equal-source base coefficient matched to
the old reference. Top-level `discrete_*_rates` contain the weighted rates:
approximately **0.32825707 and 0.16412854** for mechanical-energy coefficients.
These are prescribed source coefficients, not state-independent measurements
of net gas-energy deposition.

## Completed local checks

- 22 source/injection/budget tests pass with Python 3.11, JAX 0.10.2 CPU.
- Explicit weights (1,1) preserve the empty-weight result exactly, including
  a nonzero initial flow test.
- Weighted disjoint sources match independent single sources with individually
  scaled mass parameters, including state-dependent pressure conversion.
- At-rest discrete mass and energy increments match each source's coefficients.
- Invalid, nonpositive or mismatched weights are rejected.
- A short 32³, three-event run to t=0.02 passed online/offline comparisons:
  zero scalar difference, exact histogram counts, matching sparse surfaces,
  slices and final state. This is a smoke test, not a collision experiment.

## Remote commands, after committing and pulling the changes

Use whichever GPU the user has checked is free. Activate `.venv-iwr` and set
`CUDA_VISIBLE_DEVICES` before these commands. The user runs the GPU job.

```sh
python -m experiments.wind_bubble.check_gpu_environment
python -m unittest tests.experiments.test_unequal_source_injection tests.experiments.test_two_bubbles tests.experiments.test_unequal_source_plan tests.experiments.test_two_bubble_source_budget
python -m unittest discover -s tests/physics_modules/shock_finder -p 'test_*.py'
python -m experiments.wind_bubble.run_two_bubbles \
  --num-cells 64 --num-analysis 17 --t-end 0.16 \
  --source-strengths 1 0.5 \
  --match-source-budget outputs/two_bubbles_iwr_compgpu9_n064/metadata.json \
  --verify-offline --require-gpu \
  --output-dir outputs/two_bubbles_unequal_iwr_n064
cat outputs/two_bubbles_unequal_iwr_n064/equivalence.json
python -m experiments.wind_bubble.plot_two_bubbles \
  outputs/two_bubbles_unequal_iwr_n064/online --event-index 16 \
  --output outputs/two_bubbles_unequal_iwr_n064/event_0016.png
```

The explicit shock-finder suite is included as the established remote regression
check; local JAX 0.10.2 CPU checks do not replace the `.venv-iwr` validation.
Use a new output directory if it already exists. Acceptance checks: metadata
complete; 17 increasing actual times; stated weighted rates; online/offline
pass; endpoint flags and boundary clearance reviewed before interpreting
collision results. Then apply the frozen grouping/tracking/support rules to
the unequal catalogs. This postprocessing and its results await the GPU data;
do not presume that two equal-size families or the equal-case times will recur.
