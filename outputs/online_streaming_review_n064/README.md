# Review of the callback-generated tracking history

Open `tracking_review.png`. This uses the completed equal-source 64-cubed GPU
run in `outputs/online_streaming_iwr_compgpu14_n064`. It adds a display only;
it does not rerun simulation, grouping, matching, or surface reconstruction.

## How to read the figure

- **A to B, t = 0.07 to 0.08:** the two detected groups G00 and G01 are associated
  with a single newly labeled group G02. Each earlier group has 1,344 samples;
  the later group has 3,096 samples. Both association distances accept the
  combination. The new segment preserves that two-parent relationship.
- **C to D, t = 0.13 to 0.14:** ten small retained groups, each with 20 samples,
  are associated with two larger groups. G03 through G07 are parents of G13;
  G08 through G12 are parents of G14. Each later group has 152 samples. Both
  association distances accept both combinations. G02 continues separately.
- **Bottom:** the complete saved history has 39 group observations, 15 segments,
  24 shared continuations and three shared combinations. Each dot is one group
  observation. A segment color persists across accepted shared continuations;
  combinations begin a new segment. The complete history is shown at once,
  rather than presented as a live view with future observations hidden.

The upper row shows all detected samples at its two events. The middle row
zooms into the source neighborhoods and omits G02 so the inner groups are
visible. All 200 samples belonging to the ten groups at t = 0.13 are shown.
At t = 0.14 the view shows all 304 samples in G13/G14 and eight light-gray
samples that have no retained group label. Black stars mark the wind sources.
All positions are original refined surface samples; this figure uses no mesh
smoothing. The same colors identify groups in the spatial and history views.

Counts describe detected samples at different times and are not conserved
mass or area. Group appearance times depend on detection, spatial grouping
and the minimum retained size of 20. These times need separate validation
before interpreting them as physical shock-formation times.

These are geometric associations. The figure supports reviewing the saved
decisions, not proving unique physical ancestry. Conditional fragment-support
records remain separate and are not drawn in this overview.

## Checks and reproduction

The copied online/offline tracking products were compared again before
plotting. Per-sample labels are joined to the saved group nodes and their
counts are checked. The source catalogs, records and partitions are hashed
before rendering and checked afterward; `verification.json` also records
the plotting-source and output-image hashes. The PNG was decoded and visually
inspected for readable layout.

Run with the NumPy/SciPy/Matplotlib analysis environment and a fresh destination:

```sh
python -m experiments.wind_bubble.plot_streaming_review \
  --directory outputs/online_streaming_iwr_compgpu14_n064 \
  --output-dir outputs/online_streaming_review_n064_repeat
```
