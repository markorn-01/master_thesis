# Unequal-wind tracking at 128³

Input: `outputs/two_bubbles_unequal_iwr_compgpu1_n128/`, complete GPU run,
17 events from 0 to 0.16; source strengths 1 and 0.5. Discrete prescribed
source rates match the 64³ unequal run. There is no 128³ online/offline
reference comparison; `verification.json` records that explicitly.

Review `unequal_history.gif` (17 frames at 2 FPS), `event_0010.png` and
`event_0011.png`, then `resolution_histories.png`. The GIF shows all detected
points, a z<=0.5 cutaway, and synchronized group histories. Colors identify
continuation segments, not physical ownership. Points/normals are unchanged.
Solid graph edges are events shared by both matching distances; dashed edges
are distance-specific alternatives. Conditional family support is separate
in `fragment_windows.json` and `fragment_families.csv`, not silently inherited.

Main results with frozen settings:

- Outer groups combine over 0.09→0.10 and the combined surface continues to 0.16.
  The corresponding 64³ detected combination is 0.08→0.09.
- Seven inner fragments at 0.10 jointly cover 92.48% / 100% of the 0.11 group
  under the narrow / wide distance. Both static and delayed family tests accept
  the same parents. The 64³ early patch covered only 16.67% / 25% and failed.
- The adjacent optimizer disagrees at 0.10→0.11: narrow gate has no inner event;
  wide gate has a three-parent subset with a tiny 3.52e-8 exclusion gap.
  The conditional family is stronger geometric evidence, not proven ancestry.
- A substantial inner continuation exists over 0.11–0.16. At the final event,
  1,764 samples form this group near the stronger source. A separate 24-sample
  inward-oriented patch near the weaker source appears only at the final event.
  Two complete inner histories have not been established.
- Final outer group: 19,928 samples; 32 other samples remain below the cutoff.
  Eight upstream source overlaps are confined to those unretained samples.
  Earlier inner fragments have source overlaps; their geometry is not a
  guarantee of uncontaminated upstream state reconstruction.

Validation: readback of 17 sets of surfaces/histograms/slices; 32 independent
occupancy-DP checks; 90 independent hypothesis score checks; 16 dense family
checks; all 17 GIF frames decoded. Matching distances agree on 15/16 intervals.
29 focused unit tests pass. Source/input hashes are in `verification.json`.
No new thresholds or physical estimators were introduced.

Grouping uses 1.5 native cells and a 20-sample cutoff at both resolutions;
its physical length therefore changes. Temporal gates remain fixed in domain
units. This comparison cannot isolate tracker convergence from shock detection
and spatial grouping, and supplies no ground-truth physical ancestry.

Reproduce with a new output directory:

```sh
python -m experiments.wind_bubble.analyze_unequal_history \
  --directory outputs/two_bubbles_unequal_iwr_compgpu1_n128 \
  --output-dir outputs/two_bubbles_unequal_history_v1_n128
```

See `experiments/wind_bubble/UNEQUAL_HISTORY.md` for interpretation and scope.
