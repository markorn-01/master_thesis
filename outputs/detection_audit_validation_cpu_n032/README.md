# Local validation of selected-state detection audit

Python 3.11, JAX 0.10.2 CPU; this is a smoke test, not the 128³ physical diagnosis.
Three unchanged analysis events over t=0–0.02, unequal strengths 1:0.5,
budget matched to the saved equal 64³ reference. Selected full physical states
are saved only at events 1 and 2. The normal runner's optional observer is the
only integration change; no solver or shock-finder physics was modified.

Checks passed:

- Three ordinary events still occur; exactly two selected state files exist.
- Online/offline scalar diagnostics, histograms, catalogs, slices and final gas
  state agree (`equivalence.json`, including zero scalar difference).
- All ordinary saved products agree with a separate run without an observer
  (`audit/verification.json: reference_replay`). This comparison does not compare
  final states between the two independent runs.
- Production three-test intersection equals the zone mask on captured states.
- Separately exposed pressure/temperature jump criteria equal the combined
  production minimum-jump mask.
- Replayed surface indices and refined positions match the online catalogs exactly.
- Local source crops preserve native derivatives, source interiors and masks.
- 25 unit/regression tests passed: detection audit, unequal source injection,
  two bubbles, source-strength planning and discrete source budgets.

Baseline without capture: `outputs/detection_audit_baseline_cpu_n032/`.

Reproduce in new directories (run baseline first):

```sh
python -m experiments.wind_bubble.run_two_bubbles \
  --num-cells 32 --num-analysis 3 --t-end .02 --source-strengths 1 .5 \
  --match-source-budget outputs/two_bubbles_iwr_compgpu9_n064/metadata.json \
  --output-dir outputs/detection_audit_baseline_cpu_n032
python -m experiments.wind_bubble.audit_detection_stages \
  --num-cells 32 --num-analysis 3 --t-end .02 --capture-events 1 2 \
  --reference-run outputs/detection_audit_baseline_cpu_n032 \
  --output-dir outputs/detection_audit_validation_cpu_n032 --verify-offline
```

The remote .venv-iwr tests and 128³ audit remain pending. See
`experiments/wind_bubble/DETECTION_AUDIT.md` for the prepared GPU command.
