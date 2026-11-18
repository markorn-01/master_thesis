# Final cleanup after independent adaptive refinement

The first full three-source animation rejected event 5 (t approximately 0.05)
because the combined candidate mesh self-intersected. Its groups separately
passed the sample-agreement criterion. The original sequence and animation
remain available as records of that failure.

## Diagnosis

The rejected mesh has 34 detected intersecting pairs involving 35 distinct
faces, all between supplied groups 0 and 2 near their contact region. There
are no within-group pairs in this case. The earlier global cleanup happens
before refinement; refinement then rebuilds individual groups and cleans
them separately. Combining those refined groups can introduce cross-group
intersections that the per-group cleanup cannot see.

This is a limitation of the order of display operations. It does not establish
physical shock intersection, require merging group labels or change the
temporal ancestry definition.

## Final safety pass

`finalize_adaptive_sequence.py` reads a complete checked sequence and applies
the existing conservative defect-removal operation to events failing combined
geometry checks. This happens **after all independent refinements have been
combined**. The existing fitting/support/accuracy thresholds stay fixed.

All faces in an intersecting pair are removed, followed by the existing
nonmanifold and orientation checks. No new surface is fitted. The output
retains a small unsupported gap instead of inventing a join. The final own-group
sample agreement and global geometry checks are measured again. A repair
that loses the accuracy target is not promoted to an approved display; a
bounded cleanup failure stops the operation with the original input intact.

Surviving triangles are checked by geometry and group identity, independent
of unused-vertex compaction and triangle winding. Every output triangle must
be an exact geometric/group subset of the original candidate. All physical
arrays and group/segment labels remain exactly unchanged. Events without a
geometry failure are copied byte-for-byte, with hashes checked.

## Event-5 result

The final cleanup removes 35 of 111,664 faces (approximately 0.031%). It removes
no additional faces on the second check, requires no further winding change,
and leaves no detected self-intersection, nonmanifold edge/vertex or winding
conflict. All three groups retain 100% quarter-cell sample agreement. The
whole candidate's median and 95th-percentile sample distances are unchanged.
The original 2,872 samples are not edited or removed. Unused display vertices
may be dropped when triangle indices are compacted; retained positions do not
move. Triangle area here is a display measurement, not the physical shock-area
or thermalization estimator.

The revised sequence is `outputs/three_bubble_adaptive_sequence_final_n064/`.
Only event 5 changes; the other sixteen mesh NPZs are byte-identical to their
previous checked versions. It now has approved meshes at all sixteen populated
events, plus the empty initial state. Original points outside retained groups
remain visible as before.

`repair_0005.png` shows the local candidate, removed faces and final geometry.
`repair_0005.json` contains the pair indices, cleanup accounting and measured
group checks. Three targeted tests cover geometry-preserving keys, retention
of group identity and removal of cross-group intersections without changing an
unrelated group.

## Reproduce

Use the optional CPU environment in `surface_display_requirements.txt`:

```sh
python -m experiments.wind_bubble.finalize_adaptive_sequence \
  --directory outputs/three_bubble_adaptive_sequence_n064 \
  --output-dir outputs/three_bubble_adaptive_sequence_final_n064

python -m experiments.wind_bubble.animate_adaptive_sequence \
  --directory outputs/three_bubble_adaptive_sequence_final_n064 \
  --output-dir outputs/three_bubble_adaptive_animation_final_n064 \
  --fps 1

python -m unittest tests.experiments.test_final_combined_cleanup
```

Use fresh output directories. This corrects the remaining detected display
intersection in this saved sequence. It does not infer the true physical
contact surface, resolve uncertain tracking ancestry or the earlier same-group
two-sheet counterexample, or establish a universal reconstruction algorithm.
