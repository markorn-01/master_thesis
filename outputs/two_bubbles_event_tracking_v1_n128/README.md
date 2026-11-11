# Fixed event tracker applied to two bubbles

Read `experiments/wind_bubble/EVENT_TRACKING_APPLICATION.md` for results,
the solver-precision correction, commands and limits.

The final selected event sets agree at both tested distances for 15 of
16 intervals. The algorithm automatically identifies the t=0.07→0.08
two-group combination. The t=0.10→0.11 fragment combinations depend on
distance and have very small score gaps. These are candidate detected-group
events, not confirmed physical ancestry. Some groups remain unmatched.

The review figure includes that weak interval. `selected_events.csv`
records exclusion gaps explicitly; they are not confidence probabilities.
`results.json.gz` preserves all hypotheses and original-baseline differences.
The verification report includes all 32 exact independent DP checks and
130 independently checked candidate scores. Original data and earlier
tracking outputs remain unchanged. No new simulation was run.
