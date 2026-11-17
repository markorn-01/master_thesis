# Three-source surfaces beside their geometric histories

`tracked_surfaces.gif` shows all 17 saved events over simulation time
0–0.16 at 1 frame per second (17 seconds of playback). No new hydrodynamic
simulation or tracking fit was performed for this visualization.

Panels:
- Upper left: every original exported surface sample, colored by its local
  continuation segment. Black stars mark the three wind sources.
- Upper right: the same samples with z > 0.5 hidden to expose the interior.
  Stars remain visible for orientation. This cut changes the display only.
- Lower left/right: the verified maximum-three and experimental maximum-five
  partner histories. Node colors match the surfaces. Black cursors and rings
  mark the current event; the complete saved history stays visible.

Solid teal links are shared continuations at both matching distances. Purple
links are shared combinations. Dashed orange links are alternatives selected
at only one distance; they do not establish an inherited continuation label.
Yellow shading marks the changed interval, 0.13–0.14. Gray samples are outside
retained groups; the outer groups use muted blue-gray colors. Camera, spatial
bounds and segment colors are fixed throughout playback. Source labels S1–S3
are descriptive; source positions were not inputs to geometric association.

## Close-up of the change

`changed_interval_0013_detail.png` uses the same fixed zoom at 0.13 and 0.14.
It displays original points, including nearby samples outside the chosen
association. G labels are continuation segments, not raw per-event group IDs:

- Baseline: G06, G07, G09 → G13.
- Expanded: G06, G07, G09, G10 → G13.
- Nearby G08 remains unlinked. This is raw group 9 at event 13. Its exclusive
  coverage in the full five-parent hypothesis is 9.30%/8.14%, below the existing
  10% criterion. The display does not imply it is physically unrelated.

This close-up shows the extra geometric association and the remaining gap.
It does not establish material ancestry or justify changing the threshold.
Two other continuations in the full scene remain distance-dependent. At the
final event there are one outer group and three inward-oriented groups; 42
upstream samples overlap injection regions. Overlap flags at each event are
shown beneath the graphs without filtering or changing the measurements.

`event_0013.png`, `event_0014.png` and `event_0016.png` provide still views.

## Verification and reproduction

The renderer checks passed verification reports and recorded input/source
hashes, matching event times, identical spatial partitions, group populations
and segment assignments across modes. Per-group coordinates are checked
against original catalog indices. All 17 GIF frames and four PNGs were decoded;
input hashes were checked again after rendering. Four existing surface-label
and cutaway tests passed, and a changed segment assignment was explicitly
rejected. `verification.json` records frame populations, overlaps and hashes.

```sh
python -m experiments.wind_bubble.animate_tracked_surfaces \
  --baseline outputs/three_bubble_history_baseline_n064 \
  --candidate outputs/three_bubble_history_expanded_n064 \
  --comparison outputs/three_bubble_history_mode_comparison_n064 \
  --output-dir outputs/three_bubble_surface_history_n064 \
  --fps 1
```

Use a fresh output directory when rerunning. Rendering is CPU-only. The code
does not smooth points, interpolate missing detections, track gas particles,
or modify existing tracking choices. The maximum-five option remains an
experimental geometric association model; physical ancestry is not established.
