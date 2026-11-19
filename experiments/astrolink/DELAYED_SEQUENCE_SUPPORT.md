# One-event-delayed support beside the existing group history

## What this implementation adds

For observations A, B and C, the layer uses an accepted B→C continuation to
ask which earlier A fragments support the B surface. It reuses the existing
fragment-family equations; it does not implement probabilistic MHT, replace
the event optimizer or claim physical ancestry. The design references and
earlier isolated-window experiments remain in `TRACKLET_FRAGMENTS.md`.

The new behavior is automatic sequence integration: future seeds come from
the existing adjacent tracker, every interval is evaluated, existing events
remain unchanged, and a missing future observation has an explicit pending
status. No ground-truth continuation is passed into association.

## Rules fixed before examining the outcome

1. Derive future one-to-one continuations from accepted adjacent events. For
   real data, require a seed to be accepted at both existing matching gates.
   Synthetic configurations have one gate and use its accepted seeds.
2. Protect currently accepted A→B one-to-one continuations: neither their
   parents nor their B targets are reassociated. Existing combination parents
   remain available as family context. This permits additional small-fragment
   support without rewriting the accepted combination.
3. Compare the static B surface with the version predicted backward using the
   existing median normal-displacement estimate from B→C. The latter assumes
   approximately constant normal motion over the short window.
4. Keep the existing 60-degree signed-normal gate, 50% coverage, 0.1 family
   score margin and 0.25 distance weight. Require at least 50% union coverage
   for the family. No source location or triangle mesh determines association.
5. Keep separate channels: accepted adjacent events, conditional geometric
   support and unresolved cases. Support disagreeing with an accepted parent
   target is explicitly listed as a conflict; it does not overwrite anything.
6. A two-state window remains `pending_future`. A missing accepted future
   continuation supplies no seed; expected synthetic continuations are
   evaluation-only. There is no invented final state or arbitrary gap bridge.

The layer allows any number of supporting fragments, because support does not
require the event model's exclusive ancestry contributions. Work is bounded
to 5,000 parent/family comparisons per window. No inherited segment labels or
physical identities are added.

## Fixed evaluation

`check_delayed_sequence.py` reads and verifies
`outputs/multiframe_tracking_baseline_v1/`. It evaluates both variants on all
48 frozen synthetic configurations and all 72 frozen real interval/gate
configurations. Known development cases are not described as held-out tests.
All original adjacent events stay exactly unchanged.

`audit_delayed_support.py` independently recomputes family models, coverage,
scores, margins, assignments and union coverage with blocked all-pairs dense
distances rather than another KD tree. Blocking bounds audit memory. Position
arithmetic is double precision; saved normals retain the same once-normalized
precision as the existing family model. Float32 simulation geometry has a
specific regression test. Synthetic row reversals check categorical decisions.

The real comparison covers unequal two bubbles at 128³ through t=0.20 and
three bubbles at 64³ through t=0.16. It retains source-overlap flags on support
parents. Those flags describe sampling quality; geometry cannot validate
source-affected Mach or heating measurements. No physical ancestry truth is
available for hydro links.

For synthetic diagnostics, the report additionally evaluates a hypothetical
union of baseline links and conditional support against stipulated truth.
This exposes errors that would occur if support were mistaken for ancestry;
that union is never written into the accepted graph. Identifiable and
identical-observation controls are reported separately.

## CPU timing scope

For each real sequence, the harness performs one untimed warmup of the full
adjacent stage and both support stages, then three paired measured repeats.
The adjacent stage is shared within each repeat; each variant's combined time
is that measured adjacent time plus its own measured support time. Variant
order alternates between repeats. Results are checked outside the timers
against the archived decisions and audited support outputs. File I/O and
independent validation are excluded. These are CPU postprocessing timings,
not the earlier GPU simulation benchmarks or independent full workflow runs.

## Use and outputs

```sh
python -m unittest tests.experiments.test_delayed_support \
  tests.experiments.test_multiframe_baseline \
  tests.experiments.test_expanded_event_tracking \
  tests.experiments.test_tracklet_fragments \
  tests.experiments.test_event_history

python -m experiments.astrolink.check_delayed_sequence \
  --output-dir outputs/multiframe_support_comparison_v1
```

Use a fresh output directory. No GPU or new hydrodynamic integration is
required. The result directory contains per-case comparisons, detailed family
and link tables, both method outputs, paired timings, a comparison figure and
hash/verification reports. The original graphs, physical estimators and smooth
display outputs remain intact. Follow the frozen checkpoint's stopping rule:
report the two variants' result without automatic threshold tuning.
