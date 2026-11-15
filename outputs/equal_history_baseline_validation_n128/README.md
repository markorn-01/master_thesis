Equal-wind 128³ baseline replay through the reusable command: 17 events,
32 independent interval/distance checks. All node/segment/event choices match
the original committed baseline. Historical simulator source hashes were
verified against recorded commit fb7d1b3631a78d606829986a7468879eb6ffa315;
no historical hydrodynamics were rerun or current physics equivalence claimed.

```sh
python -m experiments.wind_bubble.track_shock_history \
  --directory outputs/two_bubbles_budget_matched_iwr_compgpu3_n128 \
  --verify-historical-sources \
  --compare-with outputs/two_bubbles_event_history_v1_n128 \
  --output-dir outputs/equal_history_baseline_validation_n128
```

Use a fresh output directory. This invocation uses the default three partners.
