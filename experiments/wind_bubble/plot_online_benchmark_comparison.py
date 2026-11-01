"""Plot reported IWR benchmark summaries without initializing JAX."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(
        'outputs/online_benchmark_iwr_comparison/reported_summary.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path(
        'outputs/online_benchmark_iwr_comparison'))
    args = parser.parse_args()
    with args.input.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    grids = sorted({int(r['num_cells']) for r in rows})
    modes = ['solver_only', 'online', 'stored_offline']
    names = ['Scheduled solver', 'Online', 'Stored + offline']
    colors = ['#7c8795', '#087e8b', '#c46428']
    data = {(int(r['num_cells']), r['mode']): r for r in rows}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout='constrained')
    x = np.arange(len(grids))
    width = .23
    metrics = [('median_workflow_seconds', 1, 'Median execution time (s)'),
               ('median_output_payload_bytes', 1e6, 'Output per trial (MB; logarithmic)'),
               ('sampled_peak_gpu_process_mib', 1024, 'Sampled process GPU peak (GiB)'),
               ('worker_peak_host_rss_bytes', 1e9, 'Process peak host RSS (GB)')]
    for panel, (ax, (field, scale, title)) in enumerate(zip(axes.flat, metrics)):
        for j, (mode, name, color) in enumerate(zip(modes, names, colors)):
            if panel == 1 and mode == 'solver_only':
                continue  # Zero output has no position on a logarithmic axis.
            values = np.array([float(data[n, mode][field])/scale for n in grids])
            positions = x + (j-1)*width
            kwargs = {}
            if panel == 0:
                low = np.array([float(data[n, mode]['min_workflow_seconds']) for n in grids])
                high = np.array([float(data[n, mode]['max_workflow_seconds']) for n in grids])
                kwargs = dict(yerr=np.array([values-low, high-values]), capsize=3,
                              error_kw={'elinewidth': 1})
            bars = ax.bar(positions, values, width, label=name, color=color, **kwargs)
            labels = [f'{v:.3f}' if panel == 1 and v < 1 else f'{v:.2f}' for v in values]
            ax.bar_label(bars, labels=labels, padding=5, fontsize=8)
        ax.set_title(f'{chr(97+panel)}) {title}', loc='left', fontsize=11)
        ax.set_xticks(x, [f'{n}³' for n in grids])
        ax.set_xlabel('Grid resolution')
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
        if panel == 1:
            ax.set_yscale('log')
            ax.set_ylim(.01, 5000)
            ax.text(.03, .96, 'Scheduled solver: 0 bytes', transform=ax.transAxes,
                    va='top', fontsize=9)
        else:
            ax.set_ylim(0, ax.get_ylim()[1]*1.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=3, frameon=False)
    fig.supxlabel('20 analysis events; median of 3 measured runs. '
                  'Compilation and one preceding warmup run excluded. Time bars: min–max.\n'
                  'Memory: whole-worker peaks including compilation and warmup. '
                  'GPU peaks are sampled. Output: compact files + uncompressed snapshots.', fontsize=9)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'png']:
        fig.savefig(args.output_dir/f'online_benchmark_comparison.{extension}', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
