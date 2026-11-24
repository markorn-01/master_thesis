# Incremental grouping and tracking inside the hydro run

## What this step changes

`--track-shocks` connects the existing bounded incremental tracker to the
ordered host analysis callback in `run_two_bubbles` and `run_three_bubbles`.
At each analysis time the callback:

1. Computes the existing shock diagnostics and builds the current sparse catalog.
2. Writes the usual histories, histograms, catalogs and background slices.
3. Passes that same in-memory catalog to the tracker, then writes its new records.

The tracker does not read saved catalogs back from disk. It retains at most two
recent geometry catalogs between events, uses a third while assessing a window,
and releases them on successful close. The writer saves and discards emitted
records; it does not accumulate complete surface or tracking histories in RAM.
The existing global scalar-history list remains in the collision writer.

The option is disabled by default. It changes neither the shock finder nor
the event schedule, source prescription or geometric association rules. This
is the existing experimental maximum-five model: connected groups use a
1.5-cell spatial radius, signed-normal agreement within 60 degrees and at
least 20 samples. Temporal association uses the existing physical distance
gates, 1.5/64 and 2.5/64 in domain units. These gates do not automatically
scale with resolution or event cadence. Full settings and source hashes are
recorded for new runs.

## Products and interpretation

Each `online/tracking/` directory contains:

| File | Contents |
| --- | --- |
| `groups_XXXX.npz` | Current partition, with one label per original sparse sample; `-1` marks an unretained sample. |
| `event_XXXX.json.gz` | Current nodes and group statistics, adjacent association results, newly available delayed-support results, sample accounting and buffer measurements. |
| `group_history.csv` | One row per retained group observation: segment label, sample counts, Mach percentiles, source-overlap counts and source fingerprints. |
| `metadata.json` | Status, settings, counters and current buffer footprint. |
| `finish.json` | Final counts and the last interval's pending support records, written only after successful close. |

When event C arrives, adjacent B-to-C relationships are available immediately.
Delayed support for A-to-B becomes available at C's actual time. These are
separate evidence channels. The final interval has no C, so its delayed
assessment remains pending. A failed run is marked failed; it does not get a
successful terminal summary.

Labels identify geometric continuation segments. Combinations, separations,
distance-sensitive alternatives and conditional support remain explicit.
Conditional support assigns no inherited identity. This integration does not
establish physical shock ancestry, ownership by a wind source, or successful
resolution of all collision fragments.

## Validation

With both `--track-shocks` and `--verify-offline`, a small run checks:

- Existing scalar histories, histograms, sparse catalogs and slices against a
  separate stored-state simulation at matching times, plus final gas-state agreement.
- Callback tracking products against a separate incremental tracker fed by the
  offline analysis. Partitions and categorical results must match exactly;
  floating diagnostics use relative/absolute tolerances of 1e-6/1e-8.
- Those offline incremental results against whole-sequence batch execution,
  including partitions, nodes, segment labels, graph events, full adjacent
  results, both delayed-support variants and terminal pending results.
- Each group's row/position fingerprints against its own saved source catalog,
  and each CSV row against its own event record.

The batch reference independently executes the existing batch assembly, rather
than replaying the incremental implementation as its oracle. Both intentionally
share the grouping and association algorithms, so this is execution equivalence,
not independent physical validation. Stored-state verification remains limited
to grids of at most 64 cubed. Only explicit verification loads the complete
reference sequence; normal online tracking keeps recent geometry only.

Focused local tests cover output equivalence, direct catalog handoff without
disk readback, bounded buffers, empty events, altered results, failure and close
behavior, and existing two/three-source regressions. A local JAX 0.6.2 CPU smoke
run is recorded in `outputs/online_streaming_cpu_n032/`.

The subsequently completed GPU comparisons are:

- [Two bubbles, 32 cubed](../../outputs/online_streaming_iwr_compgpu14_n032/equivalence.json).
- [Two bubbles, 64 cubed](../../outputs/online_streaming_iwr_compgpu14_n064/equivalence.json).
- [Three bubbles, 64 cubed](../../outputs/three_bubbles_online_tracking_iwr_compgpu8_n064/equivalence.json).

All three archives report passing primary-output, streaming-record, whole-sequence
batch, and final-state comparisons. The three-bubble archive contains 17 events
and 45 group observations. Archive names are retained; these results establish
the tested execution equivalence, not physical ancestry accuracy or validation
of every subsequent code revision. The later
[corrected-detector two-bubble check](../../outputs/opposing_front_online_iwr_compgpu3_n064/equivalence.json)
separately verifies the recorded opposing-front correction.

The [tracking benchmark](TRACKING_BENCHMARK.md) now measures the additional
workflow cost. CPU matching runs inside the host callback and adds substantial
computational work; equivalence does not establish many-bubble scalability.

## Original small-run reproduction recipe

After committing/pushing the code and pulling it into the remote checkout,
use the activated `.venv-iwr` environment and a free GPU. For the selected
compgpu14 slot 2:

```sh
export CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m unittest tests.experiments.test_streaming_products \
  tests.experiments.test_collision_streaming \
  tests.experiments.test_streaming_history \
  tests.experiments.test_delayed_support \
  tests.experiments.test_event_history \
  tests.experiments.test_group_history_table \
  tests.experiments.test_two_bubbles tests.experiments.test_three_bubbles

python -m experiments.wind_bubble.run_two_bubbles \
  --num-cells 32 --num-analysis 17 --t-end 0.16 \
  --track-shocks --verify-offline --require-gpu \
  --output-dir outputs/online_streaming_iwr_compgpu14_n032

cat outputs/online_streaming_iwr_compgpu14_n032/equivalence.json
```

Use a fresh output directory. Two sets of analysis log lines are expected:
the online run, then analysis of the stored-state reference. The independent
batch check runs afterward without repeating hydrodynamic evolution.
After this passes, the next size check is the established 64-cubed collision
setup with tracking enabled. Both sizes have already been checked in the
archives above; this recipe is retained for reproduction, not a pending task
or a claim that compgpu14 GPU 2 is currently available. The compatible Python
environment and GPU availability must be checked on the selected host.

Existing archived results retain their original source hashes. A runner-source
update does not retroactively regenerate or revalidate those results.
