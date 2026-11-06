"""Explore AstroLink on saved two-bubble shock surfaces, independently per event.

CPU postprocessing only. Raw hierarchy membership is preserved. Display colors
refer to deepest non-root groups, not physical shock labels or temporal IDs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np


def feature_arrays(positions, normals, dx, normal_weight):
    """Use common valid geometry and an explicit isotropic, cell-unit metric."""
    positions, normals = np.asarray(positions), np.asarray(normals)
    if positions.ndim != 2 or positions.shape[1] != 3 or normals.shape != positions.shape:
        raise ValueError('Positions and normals must have matching (N,3) shapes.')
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(normal_weight) or normal_weight < 0:
        raise ValueError('Require positive dx and finite nonnegative normal weight.')
    lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(positions).all(axis=1) & np.isfinite(normals).all(axis=1) & (lengths > 0)
    rows = np.flatnonzero(valid)
    p = (positions[valid].astype(float)-.5)/dx
    n = normals[valid].astype(float)/lengths[valid,None]
    return (np.column_stack((p,normal_weight*n)) if normal_weight else p), rows


def deepest_partition(ordering, intervals, ids, point_count):
    """Display-only partition; root membership alone gives label -1."""
    ordering, intervals = np.asarray(ordering), np.asarray(intervals)
    if not np.array_equal(np.sort(ordering), np.arange(point_count)):
        raise ValueError('Ordering must be a permutation of eligible input rows.')
    if intervals.shape != (len(ids),2):
        raise ValueError('Each hierarchy entry must have a start/end interval.')
    labels = np.full(point_count, -1, dtype=np.int32)
    for j in sorted(range(len(ids)), key=lambda j: str(ids[j]).count('-')):
        start,end = map(int,intervals[j])
        if not 0 <= start < end <= point_count:
            raise ValueError('Invalid hierarchy interval.')
        if str(ids[j]) != '1':
            labels[ordering[start:end]] = j
    return labels


def restore_labels(labels, permutation, catalog_rows, catalog_size):
    """Undo input shuffling and preserve all original surface-row indices."""
    result = np.full(catalog_size,-2,dtype=np.int32)
    result[np.asarray(catalog_rows)[permutation]] = labels
    return result


def plot_comparison(path, time_value, positions, sources, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes = list(results)
    fig = plt.figure(figsize=(15,8))
    for column,mode in enumerate(modes):
        result = results[mode]
        labels = result['labels']
        groups = np.unique(labels[labels >= 0])
        colors = np.tile([.65,.65,.65,.35], (len(labels),1))
        for k,group in enumerate(groups):
            colors[labels==group] = plt.get_cmap('turbo')((k+.5)/max(len(groups),1))
        axis = fig.add_subplot(2,len(modes),column+1,projection='3d')
        axis.scatter(*positions.T,c=colors,s=2,depthshade=False)
        axis.scatter(*sources.T,c='#222222',marker='*',s=75)
        axis.set(xlim=(0,1),ylim=(0,1),zlim=(0,1),xlabel='x',ylabel='y',zlabel='z',
                 title=f'{mode}\n{len(groups)} display groups, {np.mean(labels>=0):.1%} assigned')
        axis.view_init(elev=24,azim=35);axis.set_box_aspect((1,1,1))
        density_axis = fig.add_subplot(2,len(modes),len(modes)+column+1)
        ordering,intervals = result['ordering'],result['intervals']
        density = result['density'][ordering]
        density_axis.plot(np.arange(len(density)),density,color='0.4',lw=.6)
        for j,(_,end) in enumerate(intervals):
            if result['ids'][j] == '1':
                continue
            start,end = map(int,intervals[j])
            density_axis.axvspan(start,end,alpha=.1)
        density_axis.set(xlabel='AstroLink ordering',ylabel='Normalized log point density',
                         title=f"{len(intervals)-1} non-root hierarchy entries")
        density_axis.grid(alpha=.15)
    fig.suptitle(f'AstroLink exploratory grouping at t={time_value:.2f}',fontsize=17)
    fig.text(.04,.025,'Top: deepest non-root labels for display only. Gray: no non-root assignment. Stars: source locations.\nColors are local to each run, not shock identities. Bottom density describes point sampling in feature space, not gas density.',fontsize=10)
    fig.subplots_adjust(left=.04,right=.98,bottom=.14,top=.88,wspace=.22,hspace=.32)
    fig.savefig(path,dpi=130)
    plt.close(fig)


def run(args):
    from astrolink import AstroLink
    from sklearn.metrics import adjusted_rand_score
    args.output_dir.mkdir(parents=True,exist_ok=False)
    input_hashes = {}
    summaries, stability, membership_rows = [], [], []
    events_report = []
    for event_index in args.events:
        source = args.directory/f'surfaces_{event_index:04d}.npz'
        input_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        with np.load(source) as a:
            if int(a['schema_version']) != 3:
                raise ValueError('Require a multi-source surface catalog.')
            positions,normals,sources = a['positions'],a['normals'],a['source_positions']
            dx,time_value = float(a['grid_spacing']),float(a['time'])
        primary = {}
        for weight in args.normal_weights:
            mode = 'positions' if weight == 0 else f'positions_normals_w{weight:g}'
            features, catalog_rows = feature_arrays(positions,normals,dx,weight)
            if len(features) < args.k_den:
                raise ValueError('Too few eligible geometry samples for k_den.')
            labels_by_order = []
            for seed in args.order_seeds:
                permutation = np.arange(len(features)) if seed == 0 else np.random.default_rng(seed).permutation(len(features))
                clusterer = AstroLink(features[permutation], d_intrinsic=2, adaptive=0,
                                      k_den=args.k_den, S='auto', h_style=1, workers=1,
                                      working_memory=256*1024**2, verbose=0)
                start = time.perf_counter()
                clusterer.run()
                elapsed = time.perf_counter()-start
                labels = restore_labels(deepest_partition(clusterer.ordering,clusterer.clusters,clusterer.ids,len(features)),
                                        permutation,catalog_rows,len(positions))
                # The raw ordering indexes the permuted feature array. Store
                # catalog ordering too, so each raw interval maps to original rows.
                catalog_ordering = catalog_rows[permutation[np.asarray(clusterer.ordering)]]
                basename = f'event_{event_index:04d}_{mode}_order{seed}'
                np.savez_compressed(args.output_dir/f'{basename}.npz',
                                    time=time_value,positions=positions,features=features,
                                    eligible_catalog_rows=catalog_rows,input_permutation=permutation,
                                    ordering=clusterer.ordering,catalog_ordering=catalog_ordering,
                                    hierarchy_intervals=clusterer.clusters,hierarchy_ids=clusterer.ids,
                                    significances=clusterer.significances,point_log_density=clusterer.logRho,
                                    deepest_nonroot_labels=labels,normal_weight=weight,
                                    intrinsic_dimension=2,adaptive=0,k_den=args.k_den,
                                    fitted_S=float(clusterer.S),k_link=int(clusterer.k_link))
                counts = {int(k):int((labels==k).sum()) for k in np.unique(labels[labels>=0])}
                summary = dict(event_index=event_index,time=time_value,mode=mode,normal_weight=weight,
                               order_seed=seed,catalog_samples=len(positions),eligible_samples=len(features),
                               hierarchy_entries=len(clusterer.ids),nonroot_entries=len(clusterer.ids)-1,
                               display_groups=len(counts),assigned_fraction=float((labels>=0).sum()/len(features)),
                               largest_display_group_fraction=max(counts.values(),default=0)/len(features),
                               fitted_S=float(clusterer.S),k_link=int(clusterer.k_link),elapsed_seconds=elapsed)
                summaries.append(summary)
                labels_by_order.append(labels[catalog_rows])
                for j,identifier in enumerate(clusterer.ids):
                    members = catalog_ordering[slice(*map(int,clusterer.clusters[j]))]
                    centroid = positions[members].mean(axis=0)
                    membership_rows.append(dict(event_index=event_index,mode=mode,order_seed=seed,
                                                hierarchy_index=j,hierarchy_id=str(identifier),
                                                member_count=len(members),significance=float(clusterer.significances[j]),
                                                centroid_x=float(centroid[0]),centroid_y=float(centroid[1]),centroid_z=float(centroid[2])))
                if seed == args.order_seeds[0]:
                    primary[mode] = dict(labels=labels,ordering=clusterer.ordering,
                                         intervals=clusterer.clusters,ids=clusterer.ids,density=clusterer.logRho)
                print(f'event={event_index} {mode} order={seed}: nonroot={len(clusterer.ids)-1}, assigned={summary["assigned_fraction"]:.1%}',flush=True)
            if len(labels_by_order)>1:
                for j in range(1,len(labels_by_order)):
                    stability.append(dict(event_index=event_index,mode=mode,reference_order_seed=args.order_seeds[0],
                                          other_order_seed=args.order_seeds[j],
                                          deepest_partition_adjusted_rand=float(adjusted_rand_score(labels_by_order[0],labels_by_order[j]))))
        plot_comparison(args.output_dir/f'event_{event_index:04d}_comparison.png',time_value,positions,sources,primary)
        events_report.append(dict(index=event_index,time=time_value,surface_samples=len(positions)))
    for name,records in [('summary',summaries),('order_stability',stability),('hierarchy',membership_rows)]:
        if records:
            with (args.output_dir/f'{name}.csv').open('x',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in input_hashes.items())
    metadata = dict(status='complete',astrolink=importlib.metadata.version('astrolink'),
                    numpy=np.__version__,python=platform.python_version(),events=events_report,
                    normal_weights=args.normal_weights,order_seeds=args.order_seeds,k_den=args.k_den,
                    metric='squared distance = |delta position / dx|^2 + w^2 |delta unit normal|^2',
                    adaptive=0,intrinsic_dimension=2,S='auto',h_style=1,workers=1,
                    selection='common finite positions and finite nonzero normals; no Mach filtering or subsampling',
                    labels='deepest non-root hierarchy entry for display; -1 no non-root group, -2 invalid geometry',
                    temporal_tracking=False,source_ownership=False,input_hashes_unchanged=True,
                    input_sha256=input_hashes,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    timing_scope='exploratory wall times include first-call JIT and are not performance benchmarks',
                    interpretation='point-density clusters are hypotheses, not independently validated shock identities')
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--events',nargs='+',type=int,default=[4,8,16])
    parser.add_argument('--normal-weights',nargs='+',type=float,default=[0.,1.,4.])
    parser.add_argument('--order-seeds',nargs='+',type=int,default=[0,42])
    parser.add_argument('--k-den',type=int,default=20)
    args = parser.parse_args()
    if len(set(args.normal_weights)) != len(args.normal_weights) or len(set(args.events)) != len(args.events):
        parser.error('Repeated modes or events would overwrite results.')
    if args.k_den < 2 or len(set(args.order_seeds)) != len(args.order_seeds):
        parser.error('Require k-den >= 2 and distinct order seeds.')
    run(args)


if __name__ == '__main__':
    main()
