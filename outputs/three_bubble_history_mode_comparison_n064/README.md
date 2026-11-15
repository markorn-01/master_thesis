Three-source case: same saved catalogs, same spatial groups and physical
measurements, original three-partner versus experimental maximum-five mode.

Results:
- 30/32 interval/distance configurations are unchanged. Only 0.13→0.14 changes.
- A shared four-parent combination [7,8,10,13]→4 replaces the old three-parent
  set [7,8,10]→4. Coverage increases from 64.53/77.91% to 80.23/91.86% at
  narrow/wide gates. Exclusion gaps increase from ~4.89e-8/3.09e-8 to
  0.16566/0.15169. Gaps are objective margins, not probabilities.
- Both detected outer combinations remain: 0.06→0.07 joins two groups, then
  0.07→0.08 joins the remaining detected outer group. All 30 shared continuations
  persist. This is geometric evolution, not a proof of material merger ancestry.
- Distance agreement remains 15/16 intervals. Two ordinary continuations at
  0.13→0.14 remain wide-distance-only: groups 2→1 and 18→7. Their wide-gate
  child coverage is 55.29%; the expanded arity option does not resolve them.

At 0.13, five retained inward patches lie geometrically near the third source:
[7,8,9,10,13]. The chosen event includes FOUR, leaving group 9 unlinked.
The full five covers 89.53/100% of the later group, but group 9 contributes only
9.30/8.14% exclusively, below the fixed 10% requirement. Thus the rejection is
not the arity cap; no threshold was relaxed to force inclusion. The omitted
patch is not established to be physically unrelated. Its 20 samples include
16 upstream source overlaps. Source descriptors are inspection only.

Final retained scene: one outer group (6,554 samples) plus three inward-oriented
near-source groups (177,226,177). Their upstream source overlaps are 17,8,17.
This supplies final detections near all three winds but not three completely
established inner-shock ancestries or validated source-affected Mach/heating.

Checks: all original candidates preserved; spatial partition arrays, counts and
group measurements identical between modes; both graphs reassemble from their
records. All 64 independent tracking checks and recorded input/source hashes
pass. No independent physical truth labels exist, so no zero-false-link claim.

`history_comparison.png`: changed interval shaded; solid shared links, dashed
alternatives. `changed_events.json`: complete changes and coverage/source
geometry descriptors. `association_limits.json`: final scene, unresolved events,
and rejected full-five event. Raw hypotheses remain in the expanded archive.

```sh
python -m experiments.wind_bubble.compare_tracked_histories \
  --baseline outputs/three_bubble_history_baseline_n064 \
  --candidate outputs/three_bubble_history_expanded_n064 \
  --output-dir outputs/three_bubble_history_mode_comparison_n064
```

Use a fresh output directory. No further GPU run was required for this comparison.
