# Three-bubble streaming-record display preview

Open `animation/three_bubble_group_evolution.gif`. The four panels show the
checked smooth surfaces, group-relationship history, per-group Mach summaries
and original detected-sample counts. Matching colors identify the same
geometric continuation segment in each panel.

**This is a saved-catalog replay preview, not the new callback simulation.**
The physical catalogs came from the existing GPU run
`outputs/three_bubbles_iwr_compgpu7_n064`; incremental records came from the
already verified replay in `outputs/streaming_history_replay_v1/three_n064`.
The adapter and rendering ran locally on the CPU. The report's `backend: gpu`
describes the original simulation, while `origin` identifies this preview.

## What was checked

- All 45 group observations across 17 events were joined to their original
  catalog rows, positions, actual times and statistic definitions.
- Group histories, event accounting, accepted relationships, conditional
  relationships and visibility reproduce the earlier verified JSON payloads
  exactly. The group-history CSV is byte-identical to that reference.
- All checked meshes are reused unchanged. Their original samples, group joins
  and continuation-segment labels match the newly converted tables.
- The one added conditional link is hidden at event 14 and visible at event 15,
  when its future evidence first becomes available. It does not join statistic
  curves or change an inherited label. The terminal window remains pending.
- All 17 GIF frames were decoded in the correct order; playback is 17 seconds
  at 1 FPS. Empty-state and event-15 images were visually inspected.
- The focused local CPU suite passed 63 tests, including seven new adapter
  tests. Sources, input artifacts and output-image hashes matched after the run.

See `qa.json`, `tables/verification.json` and `animation/verification.json` for
the checks and provenance. These validate the display connection; physical
ancestry, turbulent-box scaling and the new three-bubble callback run are not
established by this preview.

## Reproduction and next input

```sh
python -m experiments.wind_bubble.streaming_group_demo \
  --directory outputs/three_bubbles_iwr_compgpu7_n064 \
  --replay-tracking outputs/streaming_history_replay_v1/three_n064 \
  --output-dir outputs/three_bubble_streaming_demo_preview_repeat --fps 1
```

Use a fresh directory. For the final callback-generated demonstration, run
the established three-bubble GPU setup with `--track-shocks --verify-offline`,
copy it locally, and run the same adapter without `--replay-tracking`.
Commands and the fixed completion criteria are in
`experiments/wind_bubble/ONLINE_THREE_BUBBLE_DEMO.md`.
