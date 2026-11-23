# Matched 128^3 three-bubble collision check

The simulation completed on compgpu8 GPU 6 through t=0.16 with 17 analysis
events. The five captured states correspond to events 4, 6, 8, 12 and 16.

The first wrapper invocation failed after integration because it passed a
Namespace instead of a Path to the auditor. The corrected `--audit-only`
invocation completed the production audit and physical assessment from the
existing states, without another simulation. `audit/verification.json`
records `passed: true` and `resumed_from_saved_states: true`.

The recorded physical configuration, source budgets, event times, timestep cap,
and production mask/export invariants passed their checks. The deliberate
orchestration source change is recorded separately from unchanged production
physics. This result is a resolution experiment; it is not complete
online/offline equivalence or a convergence proof.

After SCP, all five captured-state hashes and all recorded saved-primary,
reference-input and postprocessing-source hashes were verified locally.
The independent physical assessment inputs/source hashes also matched.

The completed interpretation, comparison figures, and candidate/export review
are in
[`../three_bubble_collision_resolution_review_n064_n128/README.md`](../three_bubble_collision_resolution_review_n064_n128/README.md).

`audit/` contains the actual detector masks, profiles and full states;
`physics/` contains the independent state-pair scans; `online/` contains the
17 ordinary diagnostic events. The five raw `audit/state_*.npy` files total
about 200 MiB and are deliberate validation inputs, not ordinary compact output.
