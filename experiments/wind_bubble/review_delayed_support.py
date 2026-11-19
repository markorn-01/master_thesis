"""Show newly supported fragments beside original three-time surface samples."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.astrolink.freeze_multiframe_baseline import verify_manifest


def evidence_roles(parents, accepted, additional):
    parents, accepted, additional = map(set, (parents, accepted, additional))
    if not accepted <= parents or additional != parents-accepted or accepted & additional:
        raise ValueError('Accepted and additional parents must partition the displayed family.')
    return {group: 'conditional' if group in additional else 'accepted' for group in sorted(parents)}


def group_rows(positions, labels, group, count):
    positions, labels = np.asarray(positions), np.asarray(labels)
    if positions.ndim != 2 or positions.shape[1] != 3 or labels.shape != (len(positions),):
        raise ValueError('Invalid catalog/partition shapes.')
    if not np.isfinite(positions).all():
        raise ValueError('Require finite original positions.')
    rows = np.flatnonzero(labels == group)
    if len(rows) != count or count < 1:
        raise ValueError('Group population does not match its saved history observation.')
    return rows


def load_reviews(directory):
    hashes = {}
    report = verify_manifest(directory/'verification.json', hashes)
    baseline = Path(report['frozen_baseline'])
    specs = {s['name']: s for s in json.loads((baseline/'real_cases.json').read_text())}
    archive = directory/'real_results.json.gz'
    hashes[str(archive)] = hashlib.sha256(archive.read_bytes()).hexdigest()
    with gzip.open(archive, 'rt') as stream:
        records = json.load(stream)
    additions = {}
    for record in records:
        for link in record['result']['conditional_support']:
            if not link['already_in_adjacent']:
                key = record['case'], record['parent_event'], link['child_group']
                additions.setdefault(key, set()).add(link['parent_group'])
    reviews = []
    for (case, event, child), additional in sorted(additions.items()):
        selected = [r for r in records if r['case'] == case and r['parent_event'] == event]
        if {(r['method'], r['gate_reference_cells']) for r in selected} != {
                (method, gate) for method in ('delayed_static_family', 'delayed_normal_motion_family') for gate in (1.5, 2.5)}:
            raise ValueError('Require both methods at both saved gates.')
        families, accepted_sets, new_sets = [], [], []
        for record in selected:
            result = record['result']
            family = next(f for f in result['analysis']['families'] if f['child_group'] == child)
            if not family['coverage_pass'] or result['conflicts']:
                raise ValueError('Review requires nonconflicting conditional families.')
            families.append(family)
            accepted_sets.append({p for e in result['accepted_adjacent_events']
                                  if e['kind'] == 'combination' and e['children'] == [child] for p in e['parents']})
            new_sets.append({l['parent_group'] for l in result['conditional_support']
                             if l['child_group'] == child and not l['already_in_adjacent']})
        if len({tuple(f['parent_groups']) for f in families}) != 1 or len({f['later_group'] for f in families}) != 1:
            raise ValueError('Family membership differs between methods or gates.')
        if any(s != additional for s in new_sets) or any(s != accepted_sets[0] for s in accepted_sets):
            raise ValueError('The additional/accepted memberships differ between comparisons.')
        roles = evidence_roles(families[0]['parent_groups'], accepted_sets[0], additional)
        spec = specs[case]
        graph_path = Path(spec['history'])/'graph.json'
        hashes[str(graph_path)] = hashlib.sha256(graph_path.read_bytes()).hexdigest()
        graph = json.loads(graph_path.read_text())
        later = families[0]['later_group']
        if not any(e['parent_event'] == event+1 and e['kind'] == 'continuation' and e['status'] == 'shared'
                   and e['parents'] == [f'e{event+1:04d}:g{child}']
                   and e['children'] == [f'e{event+2:04d}:g{later}'] for e in graph['events']):
            raise ValueError('The displayed future seed is not an accepted shared continuation.')
        frames = []
        for index, groups in zip(range(event, event+3), [list(roles), [child], [later]]):
            source = Path(spec['root'])/'online'/f'surfaces_{index:04d}.npz'
            partition = Path(spec['partitions'])/f'groups_{index:04d}.npz'
            for path in (source, partition):
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if str(path) in hashes and hashes[str(path)] != actual:
                    raise ValueError('Catalog/partition changed since the validated comparison.')
                hashes[str(path)] = actual
            with np.load(source) as data, np.load(partition) as saved:
                physical = {k: data[k].copy() for k in data.files}
                labels = saved['labels'].copy()
            if abs(float(physical['time'])-spec['actual_times'][index]) > 1e-7:
                raise ValueError('Catalog and history time differ.')
            observations = []
            for group in groups:
                node = next(n for n in graph['nodes'] if n['event'] == index and n['group'] == group)
                rows = group_rows(physical['positions'], labels, group, node['surface_samples'])
                upstream = int(physical['upstream_inside_any_source'][rows].sum())
                downstream = int(physical['downstream_inside_any_source'][rows].sum())
                if upstream != node['upstream_inside_source'] or downstream != node['downstream_inside_source']:
                    raise ValueError('Sampling-quality descriptors differ from saved group statistics.')
                observations.append(dict(group=group, rows=rows, samples=len(rows),
                                         upstream=upstream, downstream=downstream))
            frames.append(dict(event=index, time=float(physical['time']), physical=physical,
                               labels=labels, groups=observations))
        points = np.concatenate([f['physical']['positions'][g['rows']] for f in frames for g in f['groups']])
        pad = 2*float(frames[0]['physical']['grid_spacing'])
        low, high = points.min(axis=0)-pad, points.max(axis=0)+pad
        coverage = {str(r['gate_reference_cells']): next(f['child_union_coverage'] for f in r['result']['analysis']['families']
                    if f['child_group'] == child) for r in selected if r['method'] == 'delayed_normal_motion_family'}
        reviews.append(dict(case=case, parent_event=event, child=child, later=later,
                            roles=roles, frames=frames, low=low, high=high, coverage=coverage))
    return reviews, hashes


def render(review, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    colors = dict(accepted='#248b87', later='#397ab1', context='#c9ced2')
    additional = sorted(g for g, role in review['roles'].items() if role == 'conditional')
    extra_colors = dict(zip(additional, ['#e38a24', '#b656a1']))
    if len(additional) > 2:
        raise ValueError('This bounded review supports at most two newly supported parents.')
    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], left=.04, right=.98,
                           bottom=.16, top=.82, wspace=.10, hspace=.12)
    titles = ['A: earlier fragments', 'B: later detected group', 'C: extra observation']
    for i, frame in enumerate(review['frames']):
        ax = fig.add_subplot(grid[0, i], projection='3d')
        physical = frame['physical']; positions = physical['positions']
        selected = np.concatenate([g['rows'] for g in frame['groups']])
        local = ((positions >= review['low']) & (positions <= review['high'])).all(axis=1)
        local[selected] = False
        ax.scatter(*positions[local].T, s=3, c=colors['context'], alpha=.5, depthshade=False, linewidths=0)
        for group in frame['groups']:
            points = positions[group['rows']]
            color = extra_colors.get(group['group'], colors['accepted']) if i == 0 else colors['later']
            ax.scatter(*points.T, s=15 if i == 0 else 5, c=color, depthshade=False, linewidths=0)
            if i == 0 and group['group'] in extra_colors:
                center = points.mean(axis=0)
                ax.text(*center, f'  A:g{group["group"]}', color=color, fontsize=10, fontweight='bold')
        sources = physical['source_positions']
        sources = sources[((sources >= review['low']) & (sources <= review['high'])).all(axis=1)]
        ax.scatter(*sources.T, c='black', marker='*', s=75, depthshade=False)
        ax.set(xlim=(review['low'][0], review['high'][0]), ylim=(review['low'][1], review['high'][1]),
               zlim=(review['low'][2], review['high'][2]), xlabel='x', ylabel='y', zlabel='z',
               box_aspect=review['high']-review['low'])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
            axis.set_tick_params(labelsize=8)
        ax.view_init(22, -60)
        count = sum(g['samples'] for g in frame['groups'])
        upstream = sum(g['upstream'] for g in frame['groups'])
        ax.set_title(f'{titles[i]}\nt = {frame["time"]:.2f} | {count:,} family samples\nUpstream source-overlap flags: {upstream}', fontsize=10, pad=12)
    ax = fig.add_subplot(grid[1, :]); ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(-.8, len(review['roles'])-.15))
    middle = (len(review['roles'])-1)/2
    group_rows_map = {g['group']: g for g in review['frames'][0]['groups']}
    for y, (group, role) in enumerate(sorted(review['roles'].items(), reverse=True)):
        color = extra_colors[group] if role == 'conditional' else colors['accepted']
        row = group_rows_map[group]
        ax.scatter(.23, y, color=color, s=55, zorder=3)
        ax.text(.21, y, f'A:g{group}  ({row["samples"]} points)', ha='right', va='center', fontsize=10, color=color)
        target = (.61, middle) if role == 'conditional' else (.43, middle)
        ax.plot([.23, target[0]], [y, target[1]], color=color,
                linestyle='--' if role == 'conditional' else '-', linewidth=1.6)
    ax.scatter(.43, middle, c=colors['accepted'], marker='D', s=45, zorder=4)
    ax.plot([.43, .61], [middle, middle], c=colors['accepted'], lw=1.8)
    ax.annotate('', xy=(.88, middle), xytext=(.63, middle), arrowprops=dict(arrowstyle='->', color=colors['later'], lw=2))
    for x, prefix, frame in [(.62, 'B', review['frames'][1]), (.89, 'C', review['frames'][2])]:
        row = frame['groups'][0]
        ax.scatter(x, middle, c=colors['later'], s=95, zorder=5)
        ax.text(x, middle+.65, f'{prefix}:g{row["group"]}\n{row["samples"]:,} points', ha='center', fontsize=11)
    ax.text(.47, -.35, f'Existing {sum(r == "accepted" for r in review["roles"].values())}-parent\ncombination', ha='center', va='top', fontsize=9, color=colors['accepted'])
    ax.text(.76, middle-.7, 'Existing continuation\nprovides the later seed', ha='center', fontsize=9, color=colors['later'])
    title = 'Unequal two bubbles, 128³' if review['case'] == 'unequal_two_n128' else 'Three bubbles, 64³'
    fig.suptitle(f'{title}: what the delayed-support layer added', fontsize=16, y=.975)
    fig.text(.5, .93, f'Existing combination: {len(review["roles"])-len(additional)} fragments | '
             f'Additional support: {len(additional)} | Conditional family: {len(review["roles"])} fragments',
             ha='center', fontsize=12)
    handles = [Line2D([], [], c=colors['accepted'], label='Existing accepted relationship'),
               Line2D([], [], c='#e38a24', ls='--', label='Additional conditional support'),
               Line2D([], [], c='black', marker='*', ls='', label='Wind source')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .105), ncol=3, frameon=False)
    fig.text(.045, .06, f'Geometric union coverage: {100*review["coverage"]["1.5"]:.2f}% / {100*review["coverage"]["2.5"]:.2f}% at the narrow / wide gate; not a confidence probability.\n'
             'Original detected points in a fixed local 3D zoom; other nearby samples are gray. Colors show evidence roles, not physical ownership.\n'
             'Dashed support does not establish ancestry or change accepted IDs. Source-overlap flags remain; better grouping does not validate Mach/heating.', fontsize=9, va='center')
    name = review['case']+'_delayed_support.png'
    fig.savefig(output/name, dpi=150); plt.close(fig)
    return name


def run(directory, output):
    if output.exists():
        raise ValueError('Use a fresh review directory.')
    reviews, hashes = load_reviews(directory)
    output.mkdir(parents=True)
    descriptions = []
    for review in reviews:
        image = render(review, output)
        arrays, frames = {}, []
        for i, frame in enumerate(review['frames']):
            rows = np.sort(np.concatenate([g['rows'] for g in frame['groups']]))
            arrays[f'frame{i}_original_rows'] = rows
            arrays[f'frame{i}_positions'] = frame['physical']['positions'][rows]
            arrays[f'frame{i}_groups'] = frame['labels'][rows]
            observations = []
            for group in frame['groups']:
                observations.append({k: v for k, v in group.items() if k != 'rows'})
            frames.append(dict(event=frame['event'], time=frame['time'], observations=observations))
        artifact = output/(review['case']+'_original_points.npz')
        np.savez_compressed(artifact, **arrays)
        with np.load(artifact) as saved:
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
        descriptions.append(dict(case=review['case'], parent_event=review['parent_event'],
                                 roles=review['roles'], frames=frames, coverage=review['coverage'],
                                 bounds=[review['low'].tolist(), review['high'].tolist()],
                                 camera=[22, -60], image=image))
    (output/'review.json').write_text(json.dumps(descriptions, indent=2)+'\n')
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    report = dict(passed=True, cases=len(reviews), additional_relationships=sum(
        sum(role == 'conditional' for role in r['roles'].values()) for r in reviews),
        original_sample_positions_exact=True, labels_and_accepted_graph_unchanged=True,
        physical_ancestry_established=False, new_geometry_fitted=False,
        input_sha256=hashes, source_sha256={str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        output_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('passed', 'cases', 'additional_relationships', 'original_sample_positions_exact')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('outputs/multiframe_support_comparison_v1'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); run(args.directory, args.output_dir)
