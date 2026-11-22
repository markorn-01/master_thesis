# Completed online three-bubble demonstration

**Step 1 is complete.** Open `animation/three_bubble_group_evolution.gif`.
The movie uses the actual tracking products written during the new GPU
simulation, not the saved-catalog replay preview.

Input: `outputs/three_bubbles_online_tracking_iwr_compgpu8_n064`. The folder
retains its original compgpu8 label; execution was reported on compgpu2,
GPU 7. The 64-cubed simulation has three equal, non-collinear wind sources,
17 events through t = 0.16, and passing online/offline and batch comparisons.

## Read the four panels

1. Upper left: current checked smooth surface display, with a fixed z <= 0.5
   cutaway. Stars mark sources. Gray samples are outside approved mesh groups.
2. Upper right: geometric relationships available at the current event. Solid
   lines are shared accepted events, dotted lines are distance-specific
   alternatives, and orange dashed lines are additional conditional support.
3. Lower left: per-group median Mach and p16-p84 sample spread, calculated from
   original detections. The bands describe sample spread, not confidence bounds.
4. Lower right: original detected-sample counts per group on a logarithmic axis.
   These counts are neither surface areas nor display-triangle counts.

The same colors identify the same continuation segment across the panels.
The statistics include each complete group, including samples hidden by the
display cutaway. Combinations start new segments and separate statistic curves.

Compare `animation/event_0014.png` with `animation/event_0015.png`: the added
G08-to-G13 conditional connection becomes visible at event 15, when its C event
first exists. Accepted adjacent events remain unchanged. Conditional support
does not assign an inherited identity or merge statistic curves. The final
window has no later event, so its delayed assessment stays pending.

## Completed checks

- 45 group observations and 15 segments match the callback records.
- All 17 GIF frames decode in their expected event order; playback is 17 seconds
  at 1 FPS. Events 14 and 15 were visually reviewed for layout and link timing.
- Original sample arrays, group joins, segment labels and times match the
  previously checked mesh sequence. No mesh geometry was changed or refitted.
- The exported group CSV preserves the callback rows exactly. Local statistic
  reconstruction uses rtol 1e-6 and atol 1e-8 for floating summaries; counts,
  labels and row/position fingerprints remain exact. This accommodates tiny
  NumPy-version percentile-rounding differences without replacing saved values.
- All recorded input, code and output-image hashes match. The focused local
  CPU suite passes 65 tests, including nine adapter tests.

`qa.json`, `tables/verification.json`, and `animation/verification.json` contain
the checks and provenance. The original GPU run's `equivalence.json` and
`copy_checks.json` record simulation and copy validation.

## Reproduce the display

Use the local NumPy/SciPy/Matplotlib environment and a fresh destination:

```sh
python -m experiments.wind_bubble.streaming_group_demo \
  --directory outputs/three_bubbles_online_tracking_iwr_compgpu8_n064 \
  --output-dir outputs/three_bubble_online_tracking_demo_n064_repeat --fps 1
```

No additional GPU simulation is required. This completes the controlled
end-to-end demonstration. The separate validation summary and complete-workflow
cost measurement remain the next bounded work items. G labels and smooth display
geometry do not establish unique physical ancestry or turbulent-box scalability.
