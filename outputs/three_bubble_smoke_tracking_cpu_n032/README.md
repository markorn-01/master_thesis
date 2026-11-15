End-to-end compatibility check: the reusable five-partner command consumes the
three-source smoke catalogs, groups them and writes a checked history/overview.
Four independent interval/distance checks pass. The very early 32³ sequence
has distance-specific continuations, so this is not physical tracking validation.

```sh
python -m experiments.wind_bubble.track_shock_history \
  --directory outputs/three_bubble_validation_cpu_n032 --max-partners 5 \
  --output-dir outputs/three_bubble_smoke_tracking_cpu_n032
```

Use a new output directory. No source ownership is assigned.
