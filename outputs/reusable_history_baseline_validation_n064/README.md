Reusable command validation: default three-partner mode, 64³ unequal case,
17 saved events. All node/segment/event choices and all spatial partition
arrays reproduce the original baseline history. Thirty-two independent
geometry/optimizer checks pass. `history_overview.png` is generated automatically.

Reproduce with a fresh output directory:

```sh
python -m experiments.wind_bubble.track_shock_history \
  --directory outputs/two_bubbles_unequal_iwr_compgpu9_n064 \
  --compare-with outputs/two_bubbles_unequal_history_v1_n064 \
  --output-dir outputs/reusable_history_baseline_validation_n064
```

No experimental flag means baseline behavior. No original output is modified.
