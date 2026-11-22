"""Recheck a copied tracking benchmark and make a figure without rerunning hydro."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from experiments.wind_bubble.benchmark_tracking_workflow import (
    MODES, REFERENCE, ROOT, summarize, verify,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(directory):
    inputs = [p for p in sorted(directory.rglob('*')) if p.is_file()
              and p.relative_to(directory).parts[0] in
              ('diagnostics', 'tracked', 'benchmark.json', 'summary.csv')]
    fingerprints = {str(p.relative_to(directory)): digest(p) for p in inputs}
    benchmark = json.loads((directory/'benchmark.json').read_text())
    if benchmark['status'] != 'complete':
        raise AssertionError('Require a completed benchmark.')
    results = benchmark['modes']
    for path, expected in benchmark['source']['sha256'].items():
        if digest(ROOT/path) != expected:
            raise AssertionError(f'Current source differs from measured source: {path}')
    for mode in MODES:
        result = results[mode]
        if json.loads((directory/mode/'result.json').read_text()) != result:
            raise AssertionError('Worker result differs from controller record.')
        if digest(ROOT/REFERENCE) != result['source_budget_reference_sha256']:
            raise AssertionError('Source-budget reference differs.')
        if len({t['output_payload_bytes'] for t in result['trials']}) != 1:
            raise AssertionError('This report requires identical output sizes across trials.')
        for index, trial in enumerate(result['trials']):
            products = directory/mode/f'trial_{index:02d}'/'products'
            primary = sum(p.stat().st_size for p in products.iterdir() if p.is_file())
            tracking = sum(p.stat().st_size for p in (products/'tracking').rglob('*') if p.is_file())
            if (primary, tracking, primary+tracking) != (
                    trial['primary_output_bytes'], trial['tracking_output_bytes'],
                    trial['output_payload_bytes']):
                raise AssertionError('Copied output byte counts differ.')
            phases = sum(trial[k] for k in ('construction_seconds', 'integration_seconds',
                                           'finalization_seconds'))
            if not math.isclose(phases, trial['total_workflow_seconds'], abs_tol=1e-8):
                raise AssertionError('Execution phases do not sum to total.')
        with (directory/mode/'gpu_memory_samples.csv').open() as stream:
            samples = list(csv.DictReader(stream))
        peak = max((float(r['this_pid_memory_mib']) for r in samples), default=None)
        if len(samples) != result['gpu_memory_samples'] or peak != result['sampled_peak_gpu_process_mib']:
            raise AssertionError('GPU samples differ from reported count or peak.')
    verification = verify(directory, results)
    if verification != benchmark['verification']:
        raise AssertionError('Local verification differs from remote verification.')
    rows = summarize(results)
    if rows != benchmark['summary']:
        raise AssertionError('Recomputed summary differs.')
    with (directory/'summary.csv').open() as stream:
        saved = list(csv.DictReader(stream))
    if len(saved) != len(rows):
        raise AssertionError('Summary row count differs.')
    for actual, expected in zip(saved, rows):
        if actual != {k: str(v) for k, v in expected.items()}:
            raise AssertionError('CSV summary differs.')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ['#087f8c', '#bd5921']
    labels = ['Diagnostics', 'Diagnostics\n+ tracking']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
        ax.set_xticks([0, 1], labels)
    medians = [r['median_workflow_seconds'] for r in rows]
    axes[0].bar([0, 1], medians, color=colors, width=.55)
    axes[0].errorbar([0, 1], medians,
                    yerr=[[r['median_workflow_seconds']-r['min_workflow_seconds'] for r in rows],
                          [r['max_workflow_seconds']-r['median_workflow_seconds'] for r in rows]],
                    fmt='none', ecolor='black', capsize=5)
    for index, mode in enumerate(MODES):
        trials = results[mode]['trials']
        axes[0].scatter([index-.08, index, index+.08],
                        [t['total_workflow_seconds'] for t in trials],
                        s=15, color='black', zorder=4)
        axes[0].text(index, medians[index]+1.2, f'{medians[index]:.3f} s', ha='center')
    axes[0].set_ylim(0, max(medians)*1.22)
    axes[0].set_ylabel('Execution time (s)')
    axes[0].set_title('a) Median of three measured runs', loc='left')
    primary = [r['median_primary_output_bytes']/1e6 for r in rows]
    extra = [r['median_tracking_output_bytes']/1e6 for r in rows]
    axes[1].bar([0, 1], primary, width=.55, color='#087f8c', label='Primary diagnostics')
    axes[1].bar([0, 1], extra, bottom=primary, width=.55, color='#bd5921', label='Tracking products')
    for index, row in enumerate(rows):
        axes[1].text(index, (primary[index]+extra[index])+.12,
                     f"{row['median_output_payload_bytes']/1e6:.3f} MB", ha='center')
    axes[1].set_ylim(0, max(primary[i]+extra[i] for i in range(2))*1.35)
    axes[1].set_ylabel('Saved output per measured run (decimal MB)')
    axes[1].set_title('b) Compact output per run', loc='left')
    axes[1].legend(frameon=False, fontsize=9, loc='upper left')
    fig.suptitle('Three bubbles: additional cost of online grouping and tracking', fontsize=13)
    first = results['diagnostics']
    fig.text(.5, .035,
             f"{first['num_cells']}³ cells · {first['num_analysis']} events · "
             f"t = 0–{first['t_end']:g} · {first['device_kinds'][0]}\n"
             'Time: compilation and one warmup excluded; dots = trials, error bars = min–max. Rendering excluded.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .94))
    figure = directory/'tracking_benchmark_comparison.png'
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    a, b = rows
    report = f"""# Complete online tracking workflow: GPU benchmark

## Result

The copied benchmark passed local product revalidation. For the tested three-bubble
{first['num_cells']}-cubed case, adding grouping/tracking increased median execution
from {a['median_workflow_seconds']:.3f} s to {b['median_workflow_seconds']:.3f} s:
**{b['added_seconds_vs_diagnostics']:.3f} s added, {b['overhead_vs_diagnostics_percent']:.1f}% overhead,
{b['median_workflow_seconds']/a['median_workflow_seconds']:.2f} times the baseline duration**.
This is a substantial cost; a favorable performance ratio is not a correctness requirement.

| Measurement | Diagnostics | Diagnostics + tracking |
| --- | ---: | ---: |
| Median execution (s) | {a['median_workflow_seconds']:.6f} | {b['median_workflow_seconds']:.6f} |
| Minimum–maximum execution (s) | {a['min_workflow_seconds']:.6f}–{a['max_workflow_seconds']:.6f} | {b['min_workflow_seconds']:.6f}–{b['max_workflow_seconds']:.6f} |
| Compilation, excluded (s) | {a['compile_seconds']:.3f} | {b['compile_seconds']:.3f} |
| Primary output per run (bytes) | {a['median_primary_output_bytes']} | {b['median_primary_output_bytes']} |
| Additional tracking output (bytes) | {a['median_tracking_output_bytes']} | {b['median_tracking_output_bytes']} |
| Total output per run (decimal MB) | {a['median_output_payload_bytes']/1e6:.6f} | {b['median_output_payload_bytes']/1e6:.6f} |
| Sampled whole-worker GPU peak (MiB) | {a['sampled_peak_gpu_process_mib']} | {b['sampled_peak_gpu_process_mib']} |
| Whole-worker peak host RSS (decimal GB) | {a['worker_peak_host_rss_bytes']/1e9:.6f} | {b['worker_peak_host_rss_bytes']/1e9:.6f} |

![Execution and output comparison](tracking_benchmark_comparison.png)

## What was measured

Actual execution: {first['hostname']}, CUDA_VISIBLE_DEVICES={first['cuda_visible_devices']},
{first['device_kinds'][0]}, Python {first['python']}, JAX {first['jax']}, NumPy {first['numpy']}.
Both modes used {first['num_analysis']} analysis events through t = {first['t_end']:g}, identical three-source budgets,
and the same event-related timestep boundaries. Each mode compiled once, performed
one excluded warmup, and ran {a['repeats']} measured repetitions in its own process.

Diagnostics already includes shock finding, data transfer, histories, histograms,
sparse surface catalogs and background slices. Tracked mode adds spatial grouping,
temporal associations, both delayed-support variants, group statistics and tracking
exports. The increment measures this complete implemented package, including its
comparison-oriented support variants; it is not an isolated association-kernel time.
Neither mode writes a complete gas-state sequence. Animation rendering is excluded.

Most of the added time occurs inside synchronized integration and callbacks;
the measurements do not isolate individual tracking operations or establish a
specific bottleneck. Measured output sizes are identical across the three trials
within each mode. Tracking adds {b['median_tracking_output_bytes']/1000:.3f} kB per run
({100*b['median_tracking_output_bytes']/a['median_output_payload_bytes']:.2f}% of primary output).
Logs, benchmark reports, memory-sample tables and deleted warmup output are excluded.

## Verification and interpretation

All {a['repeats']} pairs of primary diagnostic products match using the original exact/tolerant
rules. Tracked repetitions reproduce one another, and the first tracked trial agrees
with batch assembly over {verification['batch_reference_events']} events
({verification['batch_reference_adjacent_results']} adjacent and
{verification['batch_reference_delayed_results']} delayed results). Every warmup
and measured trial records the same final-state hash and {first['trials'][0]['iterations']} solver iterations.
Tracked group-observation counts are {[t['group_observations'] for t in results['tracked']['trials']]}.
Final-state hashes were checked
remotely; full final gas arrays were not copied and are not independently rehashed here.
These are consistency checks, not independent physical ancestry labels.

The local audit rechecks copied products, worker/controller records, summary arithmetic,
phase sums, output bytes, memory samples, the source-budget reference and all recorded
code fingerprints. It does not rerun hydrodynamics or GPU timing.

The broken GPU 2 generated repeated monitor warnings although the simulation used
GPU 0. The saved monitor records contain {results['diagnostics']['gpu_memory_samples']} diagnostics
samples and {results['tracked']['gpu_memory_samples']} tracked samples, with zero recorded query
exceptions. Both peaks recompute to 454 MiB. Query exceptions do not account for
all device warnings. These are sampled exact-worker-PID allocations, not complete
GPU health measurements or guaranteed true peaks. Host RSS includes imports,
compilation and warmup; the small difference is not a tracker-only memory measurement.

The workflows ran sequentially in fixed order on a shared machine; file writes
were not synchronized to physical storage. This single resolution/cadence does
not establish general scaling. The RTX 2080 Ti results must not be compared directly
with earlier H100 timings as an algorithm-speed comparison.

## Checkpoint

Step 3's measurement and interpretation are complete for this configuration.
The implemented workflow is consistent with its references, compact on disk,
and substantially slower when the complete grouping/tracking package is enabled.
Freeze this evidence for the thesis; no additional simulation or algorithm change
is required to complete this bounded benchmark step.

Reproduce the copy check and figure from the repository root:

```sh
python -m experiments.wind_bubble.summarize_tracking_benchmark {directory.relative_to(ROOT)}
```
"""
    (directory/'README.md').write_text(report)
    if any(digest(directory/p) != expected for p, expected in fingerprints.items()):
        raise AssertionError('An original benchmark file changed during readback.')
    qa = dict(passed=True, scope='Copied-data readback; no hydro rerun or new GPU timing.',
              verification=verification, original_file_count=len(inputs),
              input_sha256=fingerprints, source_fingerprints_match=True,
              source_budget_reference_matches=True, output_byte_counts_match=True,
              phase_totals_match=True, summaries_match=True, memory_samples_match=True,
              original_files_unchanged=True,
              summary_script_sha256=digest(Path(__file__)),
              generated_sha256={p.name: digest(p) for p in (figure, directory/'README.md')})
    (directory/'copy_checks.json').write_text(json.dumps(qa, indent=2)+'\n')
    print(json.dumps(dict(passed=True, files_checked=len(inputs),
                          verification=verification, figure=str(figure)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    write_report(parser.parse_args().directory.resolve())
