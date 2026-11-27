"""Read-only equal/unequal four-source comparison at matching saved times."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.wind_bubble.compare_two_bubble_resolutions import load_checked, csv_write
from experiments.wind_bubble.streaming_products import (
    load_event, compare_products, check_batch_reference,
)


def validate_pair(equal, unequal):
    for meta in (equal, unequal):
        if meta['status'] != 'complete' or meta['num_sources'] != 4:
            raise ValueError('Require completed four-source runs.')
    fixed = ('num_cells', 'source_positions', 'injection_radius', 'wind_terminal_velocity',
             'wind_mass_loss_rate_per_source', 'injection_cell_counts', 'ambient_density',
             'ambient_pressure', 'gamma', 'domain', 'requested_times', 'actual_times',
             'solver_dt_max')
    for key in fixed:
        np.testing.assert_allclose(equal[key], unequal[key], rtol=0, atol=1e-12,
                                   err_msg=f'Fixed setup differs: {key}')
    for key in ('injection_prescription', 'surface_selection', 'histogram_weighting',
                'tracking_scope', 'source_budget_reference'):
        if equal[key] != unequal[key]:
            raise ValueError(f'Fixed setup differs: {key}')
    np.testing.assert_array_equal(equal['source_strengths'], [1, 1, 1, 1])
    np.testing.assert_array_equal(unequal['source_strengths'], [1.5, 1, .75, .75])
    for key in ('discrete_mass_rates', 'discrete_energy_rates', 'nominal_wind_luminosities'):
        np.testing.assert_allclose(sum(equal[key]), sum(unequal[key]), rtol=1e-12, atol=1e-14)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_run(root):
    meta = json.loads((root/'metadata.json').read_text())
    # Strict source check before calling the reusable historical readback helper.
    for name, expected in meta['source_sha256'].items():
        path = Path(name.split('/master_thesis/', 1)[-1])
        if sha(path) != expected:
            raise ValueError(f'Source fingerprint mismatch: {path}')
    n = len(meta['actual_times'])
    expected = {'metadata.json', 'equivalence.json'}
    for side in ('online', 'offline_reference'):
        expected.update(f'{side}/{p}' for p in
                        ('history.csv', 'tracking/metadata.json', 'tracking/finish.json', 'tracking/group_history.csv'))
        for i in range(n):
            expected.update(f'{side}/{p}_{i:04d}.npz' for p in ('surfaces', 'histograms', 'slice', 'tracking/groups'))
            expected.add(f'{side}/tracking/event_{i:04d}.json.gz')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if not expected.issubset(actual):
        raise ValueError(f'Missing files: {sorted(expected-actual)}')
    # Known local review products may already accompany a checked baseline.
    extras = actual-expected
    if any('online' in Path(p).parts or 'offline_reference' in Path(p).parts for p in extras):
        raise ValueError(f'Unexpected simulation files (possible nested copy): {sorted(extras)}')
    original = {str(root/p): sha(root/p) for p in sorted(expected)}
    meta, data, _, readback = load_checked(root)
    from experiments.wind_bubble.run_two_bubbles import compare_outputs
    writers = []
    for side in ('online', 'offline_reference'):
        with (root/side/'history.csv').open() as stream:
            rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        writers.append(SimpleNamespace(directory=root/side, rows=rows,
                       tracking=SimpleNamespace(directory=root/side/'tracking')))
    comparison = compare_outputs(*writers)
    comparison.update(check_batch_reference(root/'offline_reference', root/'offline_reference/tracking'))
    reported = json.loads((root/'equivalence.json').read_text())
    for key, value in comparison.items():
        if reported.get(key) != value:
            raise ValueError(f'Copied verification differs from local recheck: {key}')
    if reported.get('final_gas_state_matches') is not True or meta['backend'] != 'gpu':
        raise ValueError('Require reported GPU final-state agreement.')
    events = [load_event(root/'online/tracking', i) for i in range(n)]
    tracking_meta = json.loads((root/'online/tracking/metadata.json').read_text())
    event_counts = Counter((e['status'], e['kind']) for r in events for e in r['tracking_events'])
    summary = dict(group_observations=sum(len(r['nodes']) for r in events),
                   segment_count=tracking_meta['segment_count'],
                   event_counts={f'{status}:{kind}': count for (status, kind), count in event_counts.items()},
                   group_counts=[len(r['nodes']) for r in events],
                   final_group_sizes=sorted([v['surface_samples'] for v in events[-1]['nodes']], reverse=True),
                   **readback)
    # The shared reader alone does not repeat equivalence; this wrapper does.
    summary['offline_equivalence_repeated'] = True
    return dict(root=root, metadata=meta, data=data, events=events, summary=summary,
                checks=comparison, input_sha256=original)


def render(output, runs, inspect):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    colors = ['#087f8c', '#bd6124']
    names = ['Equal: 1 / 1 / 1 / 1', 'Unequal: 1.5 / 1 / 0.75 / 0.75']
    limits = {field: (min(float(r['data'][i]['background'][field].min()) for r in runs for i in inspect),
                      max(float(r['data'][i]['background'][field].max()) for r in runs for i in inspect))
              for field in ('density', 'pressure')}
    high = max(1., max(float(np.ceil(r['data'][i]['surface']['mach'][r['data'][i]['surface']['mach_valid']].max()))
                      for r in runs for i in inspect if r['data'][i]['surface']['mach_valid'].any()))
    for i in inspect:
        fig = plt.figure(figsize=(14, 9))
        grid = fig.add_gridspec(2, 3, left=.055, right=.96, bottom=.12, top=.89,
                                wspace=.35, hspace=.38)
        for row, run in enumerate(runs):
            data = run['data'][i]; s = data['surface']; p = s['positions']; good = s['mach_valid']
            ax = fig.add_subplot(grid[row, 0], projection='3d')
            im = ax.scatter(*p[good].T, c=s['mach'][good], s=1.4, alpha=.6,
                            depthshade=False, cmap='viridis', norm=Normalize(0, high))
            if (~good).any(): ax.scatter(*p[~good].T, color='.7', s=1.4)
            sources = np.asarray(run['metadata']['source_positions'])
            ax.scatter(*sources.T, marker='*', c='black', s=40)
            ax.set(xlim=(0,1), ylim=(0,1), zlim=(0,1), xlabel='x', ylabel='y', zlabel='z',
                   title=names[row]+'\nOriginal detected samples')
            ax.set_box_aspect((1,1,1)); ax.view_init(elev=24, azim=35)
            fig.colorbar(im, ax=ax, shrink=.65, pad=.06, label='Mach')
            for col, field in enumerate(('density', 'pressure'), 1):
                ax = fig.add_subplot(grid[row, col]); lo, hi = limits[field]
                im = ax.imshow(data['background'][field].T, origin='lower', extent=(0,1,0,1),
                               cmap='magma', norm=LogNorm(lo, hi))
                ax.scatter(sources[:,0], sources[:,1], marker='*', facecolors='none',
                           edgecolors='cyan', s=50)
                ax.set(xlabel='x', ylabel='y', title=f'{field.capitalize()}; z={float(data["background"]["z_coordinate"]):.4f}')
                fig.colorbar(im, ax=ax, shrink=.8, pad=.02)
        fig.suptitle(f'Four-source strength comparison | 64³ | t={float(runs[0]["data"][i]["surface"]["time"]):.2f}', fontsize=15)
        fig.text(.055,.035,'Shared camera, spatial bounds and color scales across both runs and all three selected times. Code units.\n'
                 'Hollow stars are source projections: all four sources lie outside the saved gas slice. Point clouds are not display meshes.\n'
                 'Same total prescribed input; source positions do not assign physical ownership.', fontsize=9)
        fig.savefig(output/f'event_{i:04d}_comparison.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(14,8))
    for run, name, color in zip(runs, names, colors):
        rows = run['data']; t = np.array(run['metadata']['actual_times'])
        for ax, field, label in zip(axes[0,:2], ('mach','density_jump'), ('Mach','Sampled density ratio')):
            v = np.array([[float(d['row'][f'{field}_{q}']) for q in ('p16','median','p84')] for d in rows])
            ax.plot(t,v[:,1],color=color,label=name); ax.fill_between(t,v[:,0],v[:,2],color=color,alpha=.12)
            ax.set(title=label+': median and 16–84% range',ylabel=label)
        axes[0,2].plot(t,[len(d['surface']['positions']) for d in rows],color=color,label=name)
        axes[1,0].plot(t,run['summary']['group_counts'],color=color,label=name,marker='.')
        axes[1,1].plot(t,[e['accounting']['unretained_samples'] for e in run['events']],color=color,label=name)
        for side, style in (('upstream','-'),('downstream','--')):
            axes[1,2].plot(t,[int(d['row'][f'{side}_inside_any_source']) for d in rows],color=color,ls=style,label=name.split(':')[0]+' '+side)
    for ax,title,y in ((axes[0,2],'Detected samples','Sample count'),(axes[1,0],'Retained geometric groups','Groups per observation'),
                       (axes[1,1],'Samples excluded by grouping','Sample count'),(axes[1,2],'Sampling endpoints inside sources','Flagged samples')):
        ax.set(title=title,ylabel=y)
    for ax in axes.flat:
        ax.set_xlabel('Simulation time'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Equal versus unequal sources: saved histories',fontsize=15)
    fig.text(.04,.025,'Same grid, schedule and prescribed total input. Percentile bands describe the sample population, not uncertainty.\n'
             'Groups are geometric records, not physical shock counts. Direct density ratios and advisory source-overlap flags are retained.',fontsize=9)
    fig.tight_layout(rect=(.01,.10,.99,.94)); fig.savefig(output/'histories.png',dpi=140); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for run,name,color in zip(runs,names,colors):
        h=run['data'][-1]['histogram']
        for ax,field in zip(axes,('mach','density_jump')):
            ax.stairs(h[field+'_counts'],h[field+'_edges'],color=color,label=name)
            ax.set(xlabel='Mach' if field=='mach' else 'Sampled density ratio',ylabel='Surface-cell count')
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('Final distributions | t=0.16 | one count per selected sample')
    fig.text(.04,.015,'Shared saved bins; no area weighting. Invalid and out-of-range counts are preserved in the CSV and original histories.',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.92)); fig.savefig(output/'final_histograms.png',dpi=140); plt.close(fig)
    return dict(**limits, mach=[0,high], inspected_events=inspect)


def run(equal, unequal, output):
    if output.exists(): raise ValueError('Use a fresh comparison directory.')
    runs = [checked_run(Path(p)) for p in (equal, unequal)]
    validate_pair(*(r['metadata'] for r in runs))
    settings = [json.loads((r['root']/'online/tracking/metadata.json').read_text())['settings'] for r in runs]
    if settings[0] != settings[1]: raise ValueError('Tracker settings differ.')
    for a,b in zip(runs[0]['data'],runs[1]['data']):
        for field in ('mach', 'density_jump'):
            np.testing.assert_array_equal(a['histogram'][field+'_edges'], b['histogram'][field+'_edges'])
    output.mkdir(parents=True)
    rows=[]
    for label,r in zip(('equal','unequal'),runs):
        for i,(d,e) in enumerate(zip(r['data'],r['events'])):
            rows.append(dict(case=label,event=i,**{k:float(v) for k,v in d['row'].items()},
                             retained_groups=len(e['nodes']), retained_samples=e['accounting']['retained_samples'],
                             unretained_samples=e['accounting']['unretained_samples']))
    csv_write(output/'matched_events.csv', rows)
    limits=render(output,runs,[9,13,16])
    for r in runs:
        for path,expected in r['input_sha256'].items():
            if sha(path)!=expected: raise ValueError('Original input changed during review.')
    report=dict(passed=True, fixed_setup_and_total_prescribed_input_match=True,
        source_strengths=[r['metadata']['source_strengths'] for r in runs],
        runs=[dict(root=str(r['root']),summary=r['summary'],local_checks=r['checks'],
                   checked_source_files=len(r['metadata']['source_sha256']),input_sha256=r['input_sha256']) for r in runs],
        display=limits, full_gas_states_rechecked=False,
        final_gas_state_agreement_reported_by_gpu=True, tracking_decisions_changed=False,
        scope='Read-only saved-product comparison. No physical source ownership, complete shock detection, turbulence or performance claim.',
        source_sha256={str(p):sha(p) for p in (Path(__file__),
            Path('experiments/wind_bubble/compare_two_bubble_resolutions.py'),
            Path('experiments/wind_bubble/run_two_bubbles.py'),
            Path('experiments/wind_bubble/streaming_products.py'))},
        output_sha256={str(p):sha(p) for p in output.iterdir()})
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({str(r['root']):r['summary'] for r in runs},indent=2))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--equal',type=Path,required=True)
    parser.add_argument('--unequal',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); run(args.equal,args.unequal,args.output_dir)
