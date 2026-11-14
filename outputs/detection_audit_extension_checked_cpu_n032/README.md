# CPU validation of schedule-preserving audit extension

JAX 0.10.2 CPU, 32³ smoke test; not the physical 128³ experiment.
Reference: `outputs/detection_audit_validation_cpu_n032/`, three events ending
at 0.02. Extension: four events ending at 0.03, same initial state and sources.

Checks passed:

- First three requested times are copied bit-for-bit from the reference.
- dt_max remains 0.02/3, rather than changing to 0.03/4.
- All three prefix events' surfaces, histograms and slices agree.
- The two previously captured full states agree within rtol=1e-6, atol=1e-8.
- All three selected extended states reproduce production zone masks and
  exported cell indices/refined positions.
- Separate stored-state verification passes at all four events, including
  final gas-state agreement (`equivalence.json`).
- Minimum detected-surface boundary clearance: 7.5 cells in this short smoke.
- Source/state hashes and saved-product readback checks pass.

Full states are retained locally but excluded from Git by `audit/.gitignore`.
Their hashes are recorded. No 128³ GPU result is claimed here.

Reproduce in a new output directory:

```sh
python -m experiments.wind_bubble.audit_detection_stages \
  --num-cells 32 --num-analysis 4 --t-end .03 --capture-events 1 2 3 \
  --reference-run outputs/detection_audit_validation_cpu_n032 \
  --extend-reference --verify-offline \
  --output-dir outputs/detection_audit_extension_checked_cpu_n032
```

The reference's two original full states must be present. The extension fails
before integration if an expected reference state is missing or changed, or if
its capture events are not included. The production runner's optional schedule
and dt-limit overrides default to the original behavior when omitted.
See `experiments/wind_bubble/DETECTION_EXTENSION.md` for the remote experiment.
