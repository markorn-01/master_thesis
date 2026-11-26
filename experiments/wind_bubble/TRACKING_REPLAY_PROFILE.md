# Saved-catalog tracking profile and support reuse

## Completed local result

The 17-event three-bubble 64-cubed sequence contains 66,522 sample observations
over time and 45 retained group observations. A local macOS ARM CPU replay
completed with the following unprofiled execution measurements:

| Matcher | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| Pre-change Git baseline | 11.038 s | 10.959 s | 11.077 s |
| Reused support | 5.802 s | 5.799 s | 5.995 s |

This is a 47.4% time reduction, or 1.90 times the replay throughput. All decoded
tracking products match exactly across paired runs. The original GPU tracking
archive agrees under the existing scalar tolerance. All 243 synthetic full-result
comparisons pass. Sixty-six focused unit tests pass.

In the separate instrumented replay, adjacent association accounts for 10.810
of 11.201 receive seconds before the change and 5.429 of 5.832 afterward.
Spatial grouping takes about 0.065 s in either mode; each delayed-support
variant takes about 0.13--0.14 s over the sequence. The function profile counts
64 versus 32 event selections and about 3.60 versus 1.80 million norm calls.
Instrumented times explain the bottleneck; the table uses separate unprofiled
trials. Selection costs are small compared with the repeated point search.

Evidence: `outputs/tracking_replay_optimization_n064/verification.json`,
`summary.csv`, `trials.csv`, `event_phases.csv`, `function_profile.csv` and
`tracking_replay_comparison.png`. The original 3.847/26.866-second GPU workflow
benchmark is not replaced or rescaled by this local measurement.

## Question and bounded change

Before adding more wind sources, identify the expensive part of the existing
three-bubble tracker and remove one demonstrated redundancy without changing
its geometric model. This experiment needs no new hydrodynamic integration.

The initial cProfile replay placed about 97% of tracker time in adjacent-event
matching. The maximum-five matcher first called the maximum-three matcher,
which built oriented point support and selected events. It then rebuilt the
same support and selected the final maximum-five model. The first selection
and the repeated point search were discarded work.

`event_tracking._event_candidates` now builds the existing maximum-three
candidates and returns their oriented support. `expanded_tracking` reuses that
support to enumerate four- and five-partner hypotheses, and invokes event
selection only after the final candidate list is ready. The public
maximum-three path still selects its own original candidates.

The point-query traversal, normal normalization, distance reductions,
candidate ordering, thresholds, scores, exclusivity rules, ambiguity tests,
maximum-five cap and delayed-support rules are preserved. No point thinning,
approximate search, new source labels or display geometry is introduced.

## Profiling and measurement

`profile_tracking_replay.py` loads the complete saved catalog sequence before
timing. It freezes the old `event_tracking.py` and `expanded_event_tracking.py`
from an explicit Git revision and executes those functions as the reference.
All other tracker, statistics and writer code is shared between modes.

Both modes write the normal streaming products into private temporary
directories. The execution timer includes writer construction, all event
receives and writer close, including tracking output. Decoding/comparison of
the products happens after the timer stops. Temporary trial files are then
removed; compact decoded products are retained in the result directory.

Each mode has one untimed warmup and three measured trials. Trial order
alternates baseline/optimized then optimized/baseline. These trials share a
process and allocator/filesystem caches. Writes are closed/flushed without
`fsync`. Input loading, all correctness checks, plotting and the separately
instrumented/cProfile runs are excluded from the speed comparison.

Separate phase measurements identify spatial grouping, adjacent association,
static and motion-based delayed support, geometry/table statistics, segment
assignment, copying and event assembly. The receive remainder includes
serialization, compression, file writes and writer bookkeeping. Per-event
workload records include sample counts, retained groups, spatial edges,
candidate events and high-arity hypothesis counts. Phase call totals include
the terminal pending-support calculation; per-event rows cover receives only.
Function profiles include product decoding, so their cumulative times must
not be read as independent execution benchmarks or summed across call levels.

## Correctness

The baseline and optimized paths must have exactly equal decoded products on
every warmup, measured trial and instrumented/profiled replay. This includes
partition values/dtypes, full candidate scores and selections, ambiguities,
all adjacent and delayed records, group statistics, sample fingerprints,
CSV values, buffer counters, metadata and terminal pending records. ZIP/GZIP
headers and container timestamps are not scientific outputs.

The baseline also agrees with the original saved GPU tracking archive using
the established scalar tolerances (rtol 1e-6, atol 1e-8); categories and integer
partition arrays remain exact. This archive comparison allows known NumPy
percentile-rounding differences. The new paired comparison uses exact equality.

The harness additionally compares complete old/new matcher results on the
existing synthetic suites, both adjacent intervals where available, all
declared gates and maximum partner counts 3/4/5. These are development and
regression controls, not held-out physical validation. Focused tests ensure a
five-way event uses one support tree and one final selection, and that
profiling preserves empty-event and terminal-pending behavior.

Input catalogs, original archived tracking products and all executed sources
are fingerprinted before and after the comparison. Original runs and their
historical benchmark reports are not overwritten.

## Reproduce

Use a compatible NumPy/SciPy/Matplotlib CPU environment and a fresh output path:

```sh
python -m experiments.wind_bubble.profile_tracking_replay \
  --directory outputs/three_bubbles_online_tracking_iwr_compgpu8_n064 \
  --reference-revision fab33bf10e8b6539b55faada49ddee02ab803f52 \
  --output-dir outputs/tracking_replay_optimization_n064

python -m unittest tests.experiments.test_tracking_replay_profile \
  tests.experiments.test_event_tracking \
  tests.experiments.test_expanded_event_tracking \
  tests.experiments.test_streaming_history \
  tests.experiments.test_delayed_support \
  tests.experiments.test_event_history \
  tests.experiments.test_streaming_products \
  tests.experiments.test_group_history_table \
  tests.experiments.test_tracking_benchmark
```

The result directory contains paired trial times, per-event phase/workload
tables, function-profile tables, a comparison figure, frozen reference
sources, decoded products and verification/source/input fingerprints.

## Interpretation and stopping point

This is local CPU saved-catalog performance. It excludes hydrodynamics, shock
detection, device-to-host transfer and animation/mesh rendering. Preloading
all input catalogs is convenient for the replay and is not a new production
memory policy. The online tracker still retains only recent geometry.

A replay improvement is not a measured reduction of the earlier GPU workflow
time or proof of general many-bubble scaling. The earlier 3.847/26.866-second
GPU comparison remains historical. No new physical ancestry, detection
completeness or collision-wave claim follows from exact execution agreement.

Stop after profiling this sequence, removing this one redundant calculation,
checking preserved outputs and reporting the local timing. A many-source
pilot and another full GPU benchmark are separate decisions.
