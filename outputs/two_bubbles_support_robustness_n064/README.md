# Pooled-support robustness checks

CPU postprocessing of the existing 17-event 64³ sequence. Five size cutoffs,
two spatial normal limits, two association distance gates and 16 adjacent
transitions give 320 comparisons. No production tracking behavior changed.

See the [method, findings and limitations](../../experiments/astrolink/SUPPORT_ROBUSTNESS.md).

- Lowering the cutoff retains small inner patches earlier; it does not determine
  physical shock formation time.
- Keeping isolated points adds redundant candidate links in the real sequence.
- A deliberately ambiguous parallel-sheet control accepts a stipulated wrong
  parent despite 100% pooled coverage. This control **fails**; the software
  checks correctly record that failure.

Start with `size_sensitivity.png` and `parallel_sheet_control.png`.
Detailed accepted/rejected pools, redundancy flags, raw partitions and toy
expected correspondences are retained for inspection.

The experiment supports candidate relationships, not physical identities or
source ancestry. `verification.json` distinguishes successful software checks
from the failed false-link control.
