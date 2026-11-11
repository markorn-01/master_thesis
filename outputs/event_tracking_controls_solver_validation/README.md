# Controlled rerun after solver numerical scaling

The event solver's objective is scaled by 1e6; scientific rules, scores and
thresholds are unchanged. This addresses an absolute optimization-tolerance
issue found by independent checking on the real catalogs.

All 72 comparison metric rows in `summary.csv` are byte-identical to the
original `outputs/event_tracking_controls_v1/summary.csv`. The same 36
prototype configurations pass independent dense geometry, exhaustive
conflict selection and shuffled-input checks. The original stress failures
remain. `verification.json` records the current source hashes, which the
two-bubble application runner checks before running.

See `experiments/astrolink/EVENT_TRACKING.md` and
`experiments/wind_bubble/EVENT_TRACKING_APPLICATION.md` for full scope.
