# Opposing-front correction and separate bubble recheck

The known-reference collision exposed two specific failures. With exact
conservative inputs and two-cell front separation, incoming Mach-3 fronts
were located but measured as Mach 1; outgoing Mach-1.914854 fronts produced
zone candidates but no selected surface. The November 24 correction addresses
those mechanisms. It does not relax shock-zone criteria to fill the bubble gap.

## Production changes

`_shock_surface.py` retains the existing local divergence comparison and
deterministic tie-breaking, but excludes a sampled neighbor when its normal
has negative dot product with the candidate's normal. Opposing fronts in a
joined Boolean zone no longer eliminate each other as equal backward minima.
The normal uses the same nearest-neighbor cell as the Boolean zone sample.

`_shock_zones.py` adds a shared oriented ray search for state sampling. At
quarter-cell intervals, it checks the sampled cell's normal against the
starting normal. The first negative dot product caps the ray at its preceding
search position, even if the ray has not left the joined zone. A zero normal
in uniform gas is not a reversal. The first cap persists if the direction
turns back farther along the ray. Domain bounds and the maximum exit-search
distance are retained.

The guard also covers the extra pressure-profile candidates beyond a normal
zone exit. Otherwise those candidates could still cross the other front after
the initial search stopped safely. Primary pressure-extremum candidates are
clamped to the cap. The adaptive sampler's exterior margin is also capped.
Optional plateau/extrapolation candidates beyond the cap are invalid rather
than duplicated there: repeated coordinates must not fabricate a three-point
flat plateau. The search uses a JAX loop with fixed array shapes.

Normal reversal is the overlap-handling principle described in Section 2.3.3
of [Schaal and Springel](https://arxiv.org/html/1407.4117). Our Cartesian
quarter-cell search is a specific discretization of that principle, not an
identical implementation of their Voronoi-cell ray traversal.

Compression, temperature-density alignment, local jump thresholds, normal
construction, the pressure-jump Mach relation, and the hydrodynamic integrator
are unchanged. These are analysis changes; they do not evolve a different gas
state. Future diagnostic products should record the new source revision.

## Known-reference results

The same 96-case matrix was rerun at 32^3 and 64^3. Its analytic reference
generator is unchanged. Before/after results and independently sampled
endpoints are in `outputs/colliding_shocks_correction_review_n032_n064/`.

| Exact-input case, two-cell separation | Before | After, at both resolutions |
| --- | --- | --- |
| Incoming pair, true M = 3 | Both surfaces, M = 1 | Both surfaces, M = 3 |
| Outgoing pair, true M = 1.914854 | No selected surfaces | Both surfaces, M = 1.914854 |

For the incoming pair the corrected pressure samples are P_pre = 1 and
P_post = 11. The trace reproduces the 3D Mach values using the production
dimension-independent sampler. Outgoing surface positions remain within one
cell of the exact reference positions; no claim of exact sub-cell recovery is
made for the flat compression profile.

Isolated and widely separated exact controls still pass, and all tested
stationary contacts still produce no surface. Six of the 72 incoming/outgoing
pair cases newly meet the existing location-and-5%-Mach criterion; none that
met it before cease to meet it. This is a constructed test matrix, not a
population-level accuracy estimate.

This correction does not make every close or broadened case reliable. In
particular, sub-cell separations and several synthetic broadening cases remain
unresolved or biased. The five-cell isolated controls already have about 5.7%
incoming and 12.9% outgoing Mach bias, so those remaining errors are not all
caused by overlap. No physical-convergence claim follows from the grid-scaled
separation sweep.

## Three-bubble recheck

The corrected finder and the historical finder were evaluated on the **same
ten saved gas states** at 64^3 and 128^3, at t = 0.04, 0.06, 0.08, 0.12, 0.16.
Historical selection/zone modules are loaded read-only from Git revision
`07a8ec16a32ee50a3f444008942594d68aea54f8` and verified against the frozen
benchmark hashes. Both versions run on CPU for the direct comparison.
Original state hashes are verified; no simulation is rerun.

Full-domain shock-zone masks agree exactly before and after. Their comparison
with the archived GPU crop masks also has zero differing cells in every
checked crop. In the central volume x in [0.45,0.55], y in [0.335,0.415],
z in [0.42,0.58], the counts are:

| Grid | Time | Zone candidates | Surface samples, before → after |
| --- | --- | --- | --- |
| 64^3 | 0.04 | 272 | 144 → 144 |
| 64^3 | 0.06 | 56 | 56 → 56 |
| 128^3 | 0.04 | 1200 | 464 → 464 |
| 128^3 | 0.06 | 360 | 200 → 200 |
| Both | 0.08, 0.12, 0.16 | 0 | 0 → 0 |

The central gap therefore remains. At the later checked events the region
does not enter the shock zone, so surface selection and state-sampling fixes
cannot restore it. This confirms the distinction between the controlled
overlap failures and the bubble's earlier rejection. It does **not** prove
that no physical shock is present in the rejected layer. That question still
requires independent evidence for shock/contact mixtures in this geometry.

Outputs: `outputs/three_bubble_overlap_fix_recheck_n064_n128/`, including
`central_stage_counts.csv`, `archived_backend_comparison.csv`, and
`bubble_before_after.png`. Counts refer to grid samples, not physical objects.

## Validation and reproduction

Local CPU validation is recorded in `validation.json` in the correction review
output: 60 shock-finder tests and 28 experiment tests passed (88 total).
It includes the shock-finder suite and shared-analyzer, compact-output,
sparse-output, temporal-output, and detection-audit checks. New regression
coverage includes the known two-cell failures, opposing minima along each
Cartesian axis, refined oblique sampling, the first-reversal barrier, and
plateau validity. Remote GPU validation has not been performed for this change.

From the repo root in a compatible environment, use fresh output directories:

```bash
python -m unittest discover -s tests/physics_modules/shock_finder -p 'test_*.py'

python -m experiments.shock.check_colliding_shocks \
  --output-dir outputs/colliding_shocks_corrected_new

python -m experiments.shock.review_colliding_shock_correction \
  --corrected outputs/colliding_shocks_corrected_new \
  --output-dir outputs/colliding_shocks_correction_review_new

python -m experiments.wind_bubble.recheck_collision_after_overlap_fix \
  --output-dir outputs/three_bubble_overlap_fix_recheck_new
```

The last command requires the saved 64^3 and 128^3 audit states and the
historical Git revision. It reads them without modifying the archived runs.
