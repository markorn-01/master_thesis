Equal-wind 128³ replay with explicit five-partner tracking: 17 events,
32 independent interval/distance checks. See `history_overview.png` and the
mode comparison in `outputs/equal_history_mode_comparison_n128/`.
Spatial grouping, normal, coverage, exclusivity, scores and tie rules are fixed.
Original production data and baseline history remain unchanged.

```sh
python -m experiments.wind_bubble.track_shock_history \
  --directory outputs/two_bubbles_budget_matched_iwr_compgpu3_n128 \
  --verify-historical-sources --max-partners 5 \
  --output-dir outputs/equal_history_expanded_validation_n128
```

Use a fresh output directory. The option remains experimental; partial-parent
and source-overlap limitations persist. This is an additional saved-case check,
not a blinded holdout or general turbulent-flow tracking validation.
