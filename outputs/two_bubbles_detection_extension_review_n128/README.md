# What did the extension establish?

The weaker-source inner detections grow without relaxing any production rule.

| Time | Inward near-source samples | Samples retained | Retained groups | Upstream source overlaps |
| --- | ---: | ---: | ---: | ---: |
| 0.16 | 56 | 24 | 1 | 8 |
| 0.17 | 180 | 180 | 5 | 48 |
| 0.18 | 248 | 248 | 5 | 80 |
| 0.19 | 308 | 304 | 5 | 80 |
| 0.20 | 488 | 488 | 1 | 72 |

Probe: nearest weaker source, radius <= twice the injection radius, radial
normal alignment < -0.5. This selection is used only for inspection, not
production grouping/tracking or physical source ownership.

## Detection versus grouping versus tracking

The GPU stage audit shows 248 zone/surface cells at 0.18. At 0.20 there are
512 zone cells and 488 selected surface cells. All are exported; no local
surface is removed by source exclusion. Under the unchanged spatial grouping,
the final 488 samples belong to one group. It is more extensive than the early
patches but is not demonstrated to cover a complete physical shock surface.
This is evidence that detectable structure develops over time, not proof that
no physical reverse shock existed at earlier times.

The tracker follows the original retained patch over 0.16→0.17→0.18→0.19.
Four additional patches at 0.17 are followed through 0.18 and 0.19. The outer
and stronger-source inner continuation segments also persist through 0.20.
The final 0.20 scene contains three retained groups: outer 24,600 samples,
stronger-source inward group 1,988, and weaker-source inward group 488.

At 0.19→0.20 five earlier weak-source patches face one later group. The main
tracker permits only three parents per event, so a full five-parent event
cannot be represented. At the narrow distance, candidate subsets are ambiguous
with an exclusion gap ~9.33e-9 (below the 1e-8 tie tolerance); no inner event
is accepted. At the wide distance, subset [3,4,6] is accepted with gap
~2.38e-8. This is saved as a dashed distance-specific alternative, not a shared
physical ancestry link. The three-parent cap is a representation limitation;
this audit does not establish that removing it alone would solve association.

New conditional-family checks cover eligible triples beginning at 0.15 and
0.16. They have no accepted earlier families. The final five-to-one transition
has no later 0.20→0.21 continuation to seed the existing delayed-family test;
it was not evaluated by that test. Missing future evidence is not rejection.
No extra simulation time is automatically requested to close this last link.

Quality remains relevant: 72/488 final weak-source samples have upstream
endpoints inside an injection region (none downstream). Their median sampled
Mach is about 1.438, but geometric coherence does not validate source-affected
Mach/heating reconstruction. Stronger-source inward and outer final groups
have no endpoint source overlaps. No samples were removed to improve statistics.

## Read the figures

- `weak_source_extension.png`: original near-source inward points at 0.16,
  0.18 and 0.20, plus full 3D population histories. Outer/stronger surfaces
  omitted to reveal the weaker structure. Gray means unretained; all retained
  segment colors are non-gray. Source is a black star; camera/bounds fixed.
- `weak_source_evolution.gif`: 21 frames at 2 FPS, same near-source view and
  synchronized sample-count histories. Colors identify continuation segments,
  not Mach, gas particles, material ancestry or source labels.
- `tracking_history.png`: full group-history graph. Solid connections are shared
  between distance settings; dashed are distance-specific. Dotted vertical
  line marks the previous end time 0.16. G17, the final weaker-source group,
  has no shared predecessor.

## Verification

All 21 catalogs/histograms/slices pass established readback. Original 17
partitions match the committed prefix exactly and continuation segment IDs
are unchanged. The original 32 validated tracking optimizations are reused;
eight new distance/interval optimizations pass independent exact occupancy DP
and sparse geometry/score checks. Eight new family configurations pass dense
checks. Three of four added intervals agree between distances; the last is
uncertain. All 21 GIF frames decode. Input/source hashes are saved.
33 focused grouping/tracking/review tests pass. No physics or thresholds changed.

Artifacts: partitions, `graph.json`, `tracking_results.json.gz`, group statistics,
`distance_agreement.csv`, `weak_source_history.csv`, `new_family_windows.json`,
`verification.json`, plots and animation. Raw full states remain outside Git.

## Bounded next question

We have enough detections to study the final fragment association offline.
The next algorithm test should assess five-parent versus subset support and
ambiguity on these saved data and controlled cases. It should not relax
physical detection criteria or assert ancestry because a smoother graph is
preferred. No new GPU simulation is required for that comparison.

Reproduce with a new output directory:

```sh
python -m experiments.wind_bubble.review_detection_extension \
  --output-dir outputs/two_bubbles_detection_extension_review_n128
```
