# Collision-stage tracking review: one concrete decision

**Conclusion:** preserve the existing geometric matches as support evidence.
Do not turn all accepted links into persistent identity or merger claims.
The next bounded implementation should report which links qualify as simple
continuation candidates and which require review, without changing matching.

## What was reviewed

Three saved 128³ snapshots: t=0.13, 0.14 and 0.15. Native grouping radius
1.5 cells, signed normal limit 60 degrees, sample-count cutoff 20. Reviewed
both association distances, 1.5/64 and 2.5/64 domain units (3 and 5 cells
on this 128³ grid), over the two adjacent intervals. No new simulation,
normal fitting, triangulation or physical estimator was introduced.

The xy/xz views were inspected in the existing figures. Those views use
slabs for readability; the numerical links use all 3D samples.

| Snapshot | Outer group | Left/right larger inner groups | Small facing patches |
| --- | --- | --- | --- |
| t=0.13 | G0: 18,328 samples | G1/G20: 1,164 each | G10/G11: 32 each |
| t=0.14 | G0: 19,152 | G1/G4: 1,364 each | G2/G3: 52 each |
| t=0.15 | G0: 20,480 | G1/G2: 1,576 each | No separately retained patches |

Names such as outer/inner describe the inspected geometry. Group numbers
are event-local, not physical shock IDs. The disappearance of a separately
retained patch does not establish physical shock disappearance or merger.

## Evidence table

| Interval and samples | What the saved matcher says | Review conclusion |
| --- | --- | --- |
| 0.13→0.14: three large groups | Same three targets under both distances; strong parent and child coverage, positive exclusive support | Clear **local geometric continuation candidates**, with no physical truth label |
| 0.13→0.14: two 32-sample patches | Narrow distance links them to 52-sample patches; wide distance gives tied 100% parent coverage for two targets and rejects them as ambiguous | Distance-sensitive; keep unresolved rather than choosing one physical identity |
| 0.14→0.15: outer group | One accepted target under both distances, full support | Clear local geometric continuation candidate |
| 0.14→0.15: larger inner groups | Each links to its larger next group under both distances, but each child also receives a small-patch link | Dominant continuation candidates within a multiple-parent support event; do not declare merger ancestry |
| 0.14→0.15: two 52-sample patches | Each enters a larger inner child; adds 24 exclusive child samples at narrow distance, zero at wide distance | Distance-sensitive contribution; wide-distance link is redundant support, not independent ancestry evidence |

The narrow 32→52 patch links have parent coverage 1, runner-up coverage
0.25 and dominance gap 0.75. At the wider distance, best and runner-up
coverage are both 1, giving dominance gap 0. The data justify ambiguity,
not an arbitrary target choice.

For the 52-sample parents at 0.14→0.15, each child has 1,576 samples.
The small parent's individual coverage is 8.12% (narrow) or 15.23% (wide).
Its exclusive contribution is 24/1,576 = 1.52% (narrow), or zero (wide),
because the larger parent also supports those points. Pooled acceptance
can therefore remain true even when that parent's unique evidence vanishes.

## What “clear” and “incorrect” mean here

There is no independently labeled physical ancestry for these real snapshots.
None of these real-data links can therefore be certified physically correct
or incorrect from this review. “Clear” means an unambiguous local geometric
candidate under the two reviewed distances. It is not a validated identity.

The existing controlled disappearing-nearby-sheet example does have stipulated
truth and demonstrates an incorrect extra link from the same geometric rule.
It is retained as a counterexample. This prevents interpreting strong coverage
alone as proof of physical ancestry.

`link_review.csv` contains 20 parent records (five parents × two intervals ×
two distances): eight unique-local candidates, eight distance-sensitive
records and four multiple-parent-support records. Counts include repeated
measurements at two distances, not 20 distinct physical shocks. These
advisory review labels do not modify any original acceptance decision.

## Verification

Independent per-point neighborhood queries reproduced all 20 parent best
targets, coverage scores, runner-up scores and eligibility statuses. Child
union support and exclusive contributions were checked against the saved
archive. Original catalog/archive hashes remain unchanged; references are
recorded in `verification.json`.

Existing focused support-evidence and controlled-tracking tests were rerun.
Their known false-link case remains an explicit limitation, not a passing
physical-tracking result.

## Next bounded task and stopping criterion

Add an **advisory continuation-status table** beside the existing links:

1. Unique, nonredundant one-parent/one-child support: continuation candidate.
2. Competing targets: ambiguous, with no forced identity.
3. Multiple parents: multiple-parent support, with exclusive contributions
   reported; no automatic physical merger declaration.
4. No usable support: unmatched.

Keep distance sensitivity as an additional warning, not a tuned threshold
that forces desired identities. Test these labels on the existing translation,
fragmentation, split and nearby-sheet controls, then reproduce this three-time
review. Stop there for review; no new algorithm search, rendering refinement
or higher-resolution run is required for that deliverable. Persistent track
IDs and split/merger physics remain a separate decision.

## Existing spatial figures

- [Narrow-distance 0.13→0.14](../two_bubbles_spatial_links_n064_n128/n128_fragments_narrow.png)
- [Wide-distance 0.13→0.14](../two_bubbles_spatial_links_n064_n128/n128_fragments_wide.png)
- [Redundant support at 0.14→0.15](../two_bubbles_spatial_links_n064_n128/n128_redundant_links.png)

Reproduce the numerical review in a new output directory:

```sh
python -m experiments.wind_bubble.review_collision_links --output-dir outputs/two_bubbles_collision_link_review_n128
```

CPU postprocessing with NumPy/SciPy only; no GPU or new hydrodynamic run.
