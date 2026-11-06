# Prepared 64³ / 128³ two-bubble comparison

**Status: preflight complete; 128³ GPU data not yet generated.**

The existing 64³ source prescribes a discrete mechanical-energy coefficient
of .32825707013 per source. Keeping its parameters unchanged at 128³ gives .41498408013,
26.42% more. An explicit budget-matching option reduces the 128³ mass-rate
parameter to .00791011236 while preserving velocity parameter 10, matching both
the mass rate and prescribed energy coefficient of the reference. The existing
state-dependent EI pressure conversion means this does not guarantee identical
net energy deposition during evolution; see `source_energy_semantics.json`.

`budget_audit.json` and `source_budgets.csv` record the original, unmatched and
matched budgets. Reference settings, saved times and injection/configuration
module hashes were checked. The geometry of the injection region still changes
with resolution; this preparation does not establish convergence.

See [the method and compgpu3/GPU0 commands](../../experiments/wind_bubble/TWO_BUBBLE_RESOLUTION.md).

Local validation includes 15 focused tests and a small three-event 32³
online–offline workflow check. The small run is not higher-resolution evidence.
