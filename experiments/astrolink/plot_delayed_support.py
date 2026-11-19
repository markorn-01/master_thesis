"""Readable comparison of already verified delayed-support results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.astrolink.freeze_multiframe_baseline import verify_manifest


def render(directory):
    hashes = {}
    verify_manifest(directory/'verification.json', hashes)
    with (directory/'synthetic_comparison.csv').open() as stream:
        synthetic = list(csv.DictReader(stream))
    with (directory/'real_comparison.csv').open() as stream:
        real = list(csv.DictReader(stream))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    methods = ['delayed_static_family', 'delayed_normal_motion_family']
    colors = ['#d48b37', '#21868a']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    selected = [r for r in synthetic if r['suite'] == 'three_state_development' and r['identifiable'] == 'True']
    for ax, metric, title in zip(axes[0], ['false_links', 'missed_links'],
                                ['Hypothetical false links', 'Hypothetical missed links']):
        values = [sum(int(r['baseline_'+metric]) for r in selected if r['method'] == methods[0])]
        values += [sum(int(r['hypothetical_union_'+metric]) for r in selected if r['method'] == m) for m in methods]
        bars = ax.bar(['Adjacent', 'Static support', 'Motion support'], values,
                      color=['#89929b']+colors, zorder=3)
        ax.bar_label(bars)
        ax.set(title=title, ylabel='Links (summed)', ylim=(0, max(values+[1])*1.25+1))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', alpha=.2, zorder=0)
    for ax, case, title in zip(axes[1], ['unequal_two_n128', 'three_n064'],
                                ['Unequal two bubbles, 128³', 'Three bubbles, 64³']):
        for method, color, marker, style in zip(methods, colors, ['o', 'x'], ['--', '-']):
            selected = [r for r in real if r['case'] == case and r['method'] == method]
            times = sorted({float(r['time']) for r in selected})
            values = [np.nan if any(r['window_status'] == 'pending_future' for r in selected if float(r['time']) == t)
                      else sum(int(r['additional_support_links']) for r in selected if float(r['time']) == t)
                      for t in times]
            ax.plot(times, values, marker=marker, markersize=5, linestyle=style,
                    color=color, label='Static' if method == methods[0] else 'Motion')
        ax.axvline(times[-1], color='#888888', linestyle=':', linewidth=1)
        ax.text(times[-1], .95, 'Future unavailable', transform=ax.get_xaxis_transform(),
                ha='right', va='top', color='#666666', fontsize=8)
        ax.set(title=title, xlabel='Parent simulation time', ylabel='Added support links',
               ylim=(-.15, 5 if case == 'unequal_two_n128' else 3))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle('One later observation adds fragment evidence; accepted tracking stays unchanged', fontsize=14)
    fig.text(.065, .045, 'Top: 15 identifiable development configurations. The union of support and baseline links is diagnostic, not an accepted ancestry graph.\n'
             'Bottom: two matching gates summed—4 means two relationships found at both gates. Pending final windows are not plotted as zero.', fontsize=9)
    fig.subplots_adjust(left=.065, right=.98, bottom=.15, top=.90, hspace=.36, wspace=.22)
    destination = directory/'support_comparison_readable.png'
    fig.savefig(destination, dpi=150); plt.close(fig)
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    (directory/'plot_verification.json').write_text(json.dumps(dict(
        passed=True, input_sha256=hashes,
        source_sha256={str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        output_sha256={str(destination): hashlib.sha256(destination.read_bytes()).hexdigest()},
        physical_or_tracking_results_changed=False, pending_terminal_values_masked=True), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    render(parser.parse_args().directory)
