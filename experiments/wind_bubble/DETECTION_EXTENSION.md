# Bounded extension of the weaker-source audit to t=0.20

Purpose: observe whether the weak-source inner transition becomes more clearly
detectable after 0.16 without relaxing detection, grouping or tracking rules.
This experiment is not evidence in advance that a reverse shock must appear.

## Controls

The new `--extend-reference` audit option copies every old requested time
bit-for-bit, appends times at the same cadence, and preserves the old dt_max.
For the 128³ reference, the 17 events from 0 to 0.16 remain the prefix; append
0.17, 0.18, 0.19, 0.20. Keep dt_max=0.16/17, rather than changing it to 0.20/21.
Older metadata did not explicitly record dt_max; the documented original
runner used t_end/num_analysis. New run metadata records solver_dt_max.
The history is rerun from the same initial state, not restarted from a saved
late state. Source strengths, radius, positions and prescribed budgets remain
unchanged. No clustering or physical estimator is modified.

All 17 prefix events' surfaces/histograms/slices must agree with the completed
0.16 reference. In addition, capture the five original audit states again and
compare each with its saved reference state at rtol=1e-6, atol=1e-8, then capture
0.18 and 0.20. These are selected-state/prefix checks, not proof of complete
trajectory identity or 128³ stored-state online/offline equivalence.

Seven full 128³ five-variable float32 states occupy about 294 MB. Keep them on
disk; exclude them from Git with a scoped audit/state_*.npy rule before staging
results. They are development evidence, not routine compact output.

## Remote command after committing/pushing and pulling

Use a GPU currently confirmed free. Example for compgpu1 slot 1:

```sh
cd ~/master_thesis
git pull --ff-only
source .venv-iwr/bin/activate
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m experiments.wind_bubble.check_gpu_environment
python -m unittest tests.experiments.test_detection_audit tests.experiments.test_two_bubbles tests.experiments.test_unequal_source_injection tests.experiments.test_two_bubble_source_budget tests.experiments.test_unequal_source_plan
python -m experiments.wind_bubble.audit_detection_stages \
  --num-cells 128 --num-analysis 21 --t-end 0.20 \
  --source-strengths 1 0.5 \
  --match-source-budget outputs/two_bubbles_iwr_compgpu9_n064/metadata.json \
  --capture-events 10 12 14 15 16 18 20 \
  --reference-run outputs/two_bubbles_detection_audit_iwr_n128 \
  --extend-reference --require-gpu \
  --output-dir outputs/two_bubbles_detection_extension_iwr_n128
cat outputs/two_bubbles_detection_extension_iwr_n128/audit/verification.json
```

Use a fresh output directory. The prior reference's full states must remain on
the host to obtain all five selected-state comparisons. Inspect the report:
passed=true; prefix comparison events=17; compared_captured_states=5;
all seven states captured; original_schedule_prefix_preserved=true;
extension_dt_max=0.16/17. Reported minimum surface boundary clearance must be
reviewed before physical interpretation. Clearances concern detected points,
not all gas disturbances or sampling endpoints.

The schedule does change by appending later events, so schedule_changed=true
is expected. The old prefix and timestep limit are what must remain fixed.
Metadata complete alone means simulation completion, not completed audit.

## Acceptance and next review

Compare weaker-source local jump masks, source separation and surface counts
at 0.18/0.20 with 0.14–0.16. Then apply the existing grouping rules only if the
new detections warrant it. Record missing samples and source-overlap flags.
Do not reduce thresholds or invent an early history to make the result look
complete. If the prefix replay fails, investigate before interpreting later
measurements. No automatic increase to longer times or higher resolution.
