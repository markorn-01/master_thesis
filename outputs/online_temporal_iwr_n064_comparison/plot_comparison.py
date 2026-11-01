"""Reproduce the 64^3 temporal comparison from preserved online CSV files."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT.parent
RUNS = [('5 events', 'online_temporal_iwr_compgpu3_n064', '#cc6b28', 's'),
        ('21 events', 'online_temporal_iwr_compgpu3_n064_cadence21', '#07818c', 'o')]

def rows(folder, name):
    with (folder/'online'/f'{name}.csv').open() as f:
        return list(csv.DictReader(f))

def values(records, key):
    return np.array([float(r[key]) for r in records])

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 200})
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
summary = {}
for label, folder, color, marker in RUNS:
    folder = OUTPUTS/folder
    tracks = [r for r in rows(folder, 'tracks') if r['shock_kind'] == 'reverse']
    energy = rows(folder, 'energy')
    t = values(tracks, 'time')
    et = values(energy, 'time')
    common = dict(color=color, marker=marker, markersize=4, linewidth=1.5, label=label)
    axes[0, 0].plot(t, values(tracks, 'radius_median'), **common)
    axes[0, 1].plot(t, values(tracks, 'median_radial_alignment'), **common)
    axes[1, 0].plot(et, values(energy, 'reverse_dissipation_rate'), **common)
    e = values(energy, 'reverse_cumulative_dissipated_energy')
    axes[1, 1].plot(et, e, **common)
    detected = [r for r in tracks if r['detected'] == 'True']
    # Common-window values are useful context, not a pure quadrature comparison.
    i = int(np.argmin(abs(et - .15)))
    assert np.isclose(et[i], .15)
    summary[label] = dict(first_detection_time=float(detected[0]['time']),
        detected_events=len(detected), final_radius=float(detected[-1]['radius_median']),
        final_cumulative_energy=float(e[-1]), energy_from_015_to_020=float(e[-1]-e[i]),
        integrated_duration=float(energy[-1]['reverse_integrated_duration']))
axes[0, 0].axhline(.0625, color='0.5', ls=':', lw=1)
axes[0, 0].text(.109, .064, 'Injection radius', color='0.4', fontsize=9)
axes[0, 1].axhline(-1, color='0.5', ls=':', lw=1)
axes[0, 1].set_ylim(-1.005, -.965)
titles = ['a) Reverse-candidate radius', 'b) Median normal alignment',
          'c) Thermalization rate', 'd) Energy since first finite power sample']
ylabels = ['Radius (code units)', r'$\mathrm{median}(\hat{\boldsymbol{n}}\cdot\hat{\boldsymbol{r}})$',
           'Power (code units)', 'Accumulated energy (code units)']
for ax, title, ylabel in zip(axes.flat, titles, ylabels):
    ax.set_title(title, loc='left', fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xlim(.108, .204)
    ax.grid(alpha=.2)
for ax in axes[1]:
    ax.set_xlabel('Simulation time (code units)')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axes[0, 0].legend(loc='upper left', frameon=False)
fig.suptitle(r'Reverse-shock candidate at $64^3$: two analysis schedules', fontsize=14)
fig.text(.5, .045, 'No reverse candidate before t = 0.15 (5 events) or t = 0.12 (21 events). Energy origins differ.', ha='center', fontsize=9)
fig.text(.5, .022, 'Schedules also affect solver steps; this comparison does not isolate quadrature error or prove physical convergence.', ha='center', fontsize=9)
fig.tight_layout(rect=(0, .075, 1, .95))
for suffix in ('png', 'pdf'):
    fig.savefig(ROOT/f'reverse_candidate_cadence_comparison.{suffix}')
(ROOT/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
