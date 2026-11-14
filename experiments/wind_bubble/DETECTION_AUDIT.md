# Targeted unequal-wind detection audit

Question: why are too few inward-oriented surface samples exported near the
weaker source before t=0.15? The previous saved-data audit cannot distinguish
physical late formation, injection-region exclusion, numerical broadening,
criterion rejection or surface selection. This experiment records the missing
evidence; it does not assume a cause or change the finder.

## What changes

`run_two_bubbles.run` accepts an optional JAX-side `state_observer`. With none,
the ordinary runner behaves as before. The audit observer writes only selected
unpadded physical states, after ordinary diagnostics at the same event. It
never writes into the solver state, adds no analysis events, and does not
change timestep limits or source prescriptions. A JAX conditional prevents
transferring full states at unselected times. The GPU run keeps all 17 events.

Default captures: events 10, 12, 14, 15, 16 (t=0.10, 0.12, 0.14, 0.15, 0.16).
Five five-variable float32 128³ states occupy approximately 210 MB, plus local
compressed diagnostic crops and ordinary online output. This is intentionally
larger development/validation output, not a new routine compact-storage claim.

After integration the audit replays the unchanged production compression,
gradient-alignment, minimum-jump, zone and surface functions on each captured
state. It exposes separate pressure/temperature jump tests and asserts their
intersection exactly equals the production minimum-jump test. It also asserts
that the combined three criteria equal the production zone mask. Reconstructed
refined positions and exported surface cell indices must match the online
catalog exactly. Source strengths do not enter the finder-stage replay; grid
geometry and variable registry come from the same problem builder.

For both sources, save all primitive variables and stage fields in a local box
extending two injection radii, including the injection interior. Report counts
for the full local sphere, its inward-oriented subset, and both subsets outside
the discrete injection mask. These are diagnostic regions, not shock ownership
or physical labels. The inward filter is radial normal dot product < -0.5.

Independent criterion counts are not a causal ranking. Joint criterion-pattern
counts preserve overlaps of failures. The three-bit pattern is ordered
(minimum jump, gradient alignment, compression), e.g. 011 passes compression
and gradient alignment but fails minimum jump. 111 is a zone cell. Counts of
zone cells not selected as surfaces and surfaces removed by export identify
later stages. They do not by themselves prove a physically valid shock was
missed.

Distinguish the actual injection mask (cell center radius <= R-dx/2) from the
surface-export exclusion (refined position radius <= R around either source).
Both are retained explicitly in crop files. No exclusion is relaxed.

## Run remotely after committing/pushing and pulling

Use a GPU currently confirmed free. Example using the previously selected
compgpu1 slot 1:

```sh
cd ~/master_thesis
git pull --ff-only
source .venv-iwr/bin/activate
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m experiments.wind_bubble.check_gpu_environment
python -m unittest tests.experiments.test_detection_audit tests.experiments.test_unequal_source_injection tests.experiments.test_two_bubbles tests.experiments.test_unequal_source_plan tests.experiments.test_two_bubble_source_budget
python -m experiments.wind_bubble.audit_detection_stages \
  --num-cells 128 --num-analysis 17 --t-end 0.16 \
  --source-strengths 1 0.5 \
  --match-source-budget outputs/two_bubbles_iwr_compgpu9_n064/metadata.json \
  --capture-events 10 12 14 15 16 \
  --reference-run outputs/two_bubbles_unequal_iwr_compgpu1_n128 \
  --require-gpu \
  --output-dir outputs/two_bubbles_detection_audit_iwr_n128
cat outputs/two_bubbles_detection_audit_iwr_n128/audit/verification.json
```

Use a new output directory. Do not use `--verify-offline` at 128³; the ordinary
runner restricts it to 64³. `--reference-run` checks all ordinary saved replay
products against the prior GPU run, with exact discrete fields/histograms and
floating tolerance rtol=1e-6, atol=1e-8. It does not compare complete final gas
states or establish 128³ full-state online/offline equivalence. A replay
mismatch must be investigated before interpreting the stage audit.

## Outputs and interpretation

- Ordinary `metadata.json` and `online/` products retain their normal meaning.
- `audit/state_*.npy`: selected full primitive states, including source interiors.
- `audit/stages_*_source*.npz`: local primitive fields, grid centers, masks,
  normals, offsets, divergence and measured local log jumps. Centers/arrays
  retain their native grid alignment; no local gradients are recomputed from
  truncated crops.
- `audit/stage_counts.csv`: per-event/source/region rejection accounting.
- `audit/verification.json`: selected events, source/state hashes, production
  mask/export checks, and separate prior-run replay agreement.

Simulation metadata can say complete before offline audit completion; accept
this experiment only if `audit/verification.json` exists and says passed.

Next review compares gas profiles and masks in the weaker-source transition.
If zone rejection dominates, inspect the actual failed conditions and measured
jumps. If zones exist but surfaces disappear, inspect compression selection.
If surfaces exist inside the export exclusion, assess source/transition
separation. Only then choose a controlled modification or a longer run.
No parameters are changed to force two inner histories.
