Opt-in maximum-five-partner tracking of the same three-source GPU catalogs.
17 events, 32 independent geometry/optimizer checks; distance agreement 15/16.
The maximum is a cap, not a required number of parents. One four-parent event
is selected consistently; no five-parent association is forced.

```sh
python -m experiments.wind_bubble.track_shock_history \
  --directory outputs/three_bubbles_iwr_compgpu7_n064 --max-partners 5 \
  --output-dir outputs/three_bubble_history_expanded_n064
```

Use a fresh output directory. The method remains experimental; no physical
ancestry labels or completed many-bubble tracking are claimed.
