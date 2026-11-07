"""Observational three-event prediction audit; no physical identity ground truth."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from experiments.astrolink.motion_prediction import predict_positions
from experiments.astrolink.expansion_prediction import predict_expanding_positions
from experiments.astrolink.support_evidence import support_evidence
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write
from experiments.wind_bubble.compare_resolution_associations import support_summary


def distance_summary(values):
    v=np.asarray(values,float);finite=v[np.isfinite(v)]
    q=np.percentile(finite,[50,84,95]) if len(finite) else [None]*3
    return dict(samples=len(v),finite_samples=len(finite),median=q[0],p84=q[1],p95=q[2])


def nearest_distances(points,targets):
    """Unrestricted nearest position only: not a physical or normal-aware match."""
    if not len(targets):return np.full(len(points),np.inf)
    if not len(points):return np.empty(0)
    return cKDTree(targets).query(points,k=1)[0]


def run(directory,output):
    meta=json.loads((directory/'metadata.json').read_text())
    if not json.loads((directory/'readback.json').read_text())['passed']:raise ValueError('Require checked input.')
    hashes=dict(meta['input_sha256'])
    for name in ['metadata.json','readback.json','support_records.json.gz']:
        path=directory/name;hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for p,h in hashes.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError(f'Changed input: {p}')
    with gzip.open(directory/'support_records.json.gz','rt') as f:reference_records=json.load(f)
    root=Path(meta['target_directory']);run_meta=json.loads((root/'metadata.json').read_text())
    if run_meta['num_cells']!=128:raise ValueError('Require checked 128³ target.')
    data=[]
    for i in range(len(run_meta['actual_times'])):
        with np.load(root/'online'/f'surfaces_{i:04d}.npz') as s:surface={k:s[k].copy() for k in s.files}
        with np.load(Path(meta['comparison_directory'])/f'groups_n128_event{i:04d}.npz') as g:groups={k:g[k].copy() for k in g.files}
        data.append((surface,groups))
    output.mkdir(parents=True,exist_ok=False);records=[];summaries=[];group_rows=[];baseline_checks=0
    for middle in range(1,len(data)-1):
        surfaces=[data[i][0] for i in [middle-1,middle,middle+1]];times=[float(s['time']) for s in surfaces]
        p0,p1,p2=[s['positions'] for s in surfaces];n0,n1,n2=[s['normals'] for s in surfaces];dx=float(surfaces[1]['grid_spacing'])
        for mode,cutoff in [('native_cells',20),('native_cells_scaled_count',80)]:
            key=f'{mode}_a60_labels';l0,l1,l2=[data[i][1][key] for i in [middle-1,middle,middle+1]]
            for gate_ref in [1.5,2.5]:
                gate=gate_ref/64
                translation,te,seed=predict_positions(p0,n0,l0,p1,n1,l1,times,gate)
                expansion,ee,_=predict_expanding_positions(p0,n0,l0,p1,n1,l1,times,gate)
                config=dict(middle_event=middle,target_event=middle+1,times=times,mode=mode,cutoff=cutoff,
                            gate_reference_cells=gate_ref,gate_code_units=gate)
                for method,points,estimates in [('geometry',p1,te),('translation',translation,te),('expansion',expansion,ee)]:
                    r=support_evidence(points,n1,l1,p2,n2,l2,gate)
                    if method=='geometry':
                        ref=next(x['result'] for x in reference_records if x['config']['num_cells']==128 and x['config']['mode']==mode
                                 and x['config']['parent_event']==middle and x['config']['gate_reference_cells']==gate_ref and x['config']['segmentation_angle']==60)
                        for field in ['parents','children','links']:assert r[field]==ref[field]
                        baseline_checks+=1
                    fwd=nearest_distances(points[l1>=0],p2[l2>=0]);rev=nearest_distances(p2[l2>=0],points[l1>=0])
                    full_forward=np.full(len(p1),np.inf);full_forward[l1>=0]=fwd
                    group_medians=[];groups=[]
                    for group in np.unique(l1[l1>=0]):
                        mask=l1==group;stats=distance_summary(full_forward[mask]);estimate=next(e for e in estimates if e['group']==group)
                        previous=estimate['predecessor_group'];before=int((l0==previous).sum()) if previous is not None else None
                        if stats['median'] is not None:group_medians.append(stats['median']/dx)
                        row=dict(middle_event=middle,target_event=middle+1,target_time=times[2],mode=mode,cutoff=cutoff,
                                 gate_reference_cells=gate_ref,method=method,group=int(group),samples=int(mask.sum()),
                                 initialized=estimate['initialized'],predecessor_group=previous,predecessor_samples=before,
                                 sample_count_ratio=int(mask.sum())/before if before else None,
                                 upstream_source_overlap=int(surfaces[1]['upstream_inside_any_source'][mask].sum()),
                                 nearest_median_cells=stats['median']/dx if stats['median'] is not None else None,
                                 nearest_p84_cells=stats['p84']/dx if stats['p84'] is not None else None,
                                 nearest_p95_cells=stats['p95']/dx if stats['p95'] is not None else None)
                        group_rows.append(row);groups.append(row)
                    scalar=dict(middle_event=middle,target_event=middle+1,target_time=times[2],mode=mode,cutoff=cutoff,gate_reference_cells=gate_ref,
                                method=method,initialized_groups=sum(e['initialized'] for e in estimates),fallback_groups=sum(not e['initialized'] for e in estimates),
                                expansion_initialized_groups=sum(e.get('expansion_initialized',False) for e in estimates),
                                current_retained_samples=int((l1>=0).sum()),future_retained_samples=int((l2>=0).sum()),
                                current_excluded_samples=int((l1<0).sum()),future_excluded_samples=int((l2<0).sum()),
                                median_group_nearest_distance_cells=float(np.median(group_medians)) if group_medians else None,
                                forward_nearest=distance_summary(fwd),reverse_nearest=distance_summary(rev),**support_summary(r))
                    records.append(dict(config=config,method=method,estimates=estimates,
                                        bootstrap={k:seed[k] for k in ['parents','children','links']},
                                        result={k:r[k] for k in ['parents','children','links']},group_diagnostics=groups,summary=scalar))
                    summaries.append({k:v for k,v in scalar.items() if k not in ['forward_nearest','reverse_nearest']})
        print(f'Audited events {middle-1:02d}, {middle:02d} -> {middle+1:02d}',flush=True)
    csv_write(output/'summary.csv',summaries);csv_write(output/'group_diagnostics.csv',group_rows)
    with gzip.open(output/'results.json.gz','wt') as f:json.dump(records,f,separators=(',',':'),allow_nan=False)
    plot(output,summaries)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    sources=[Path(__file__)]+[Path('experiments/astrolink')/f'{name}.py' for name in ['motion_prediction','expansion_prediction','support_evidence','patch_support']]
    report=dict(configurations=len(records),geometry_baselines_reproduced=baseline_checks,input_sha256=hashes,input_hashes_unchanged=True,
                temporal_comparison_directory=str(directory),target_directory=str(root),partition_directory=meta['comparison_directory'],
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                description='One-step observational prediction from two earlier events; reset/bootstrap each triplet, not recursive tracking.',
                gates_reference_cells=[1.5,2.5],spatial_normal_limit=60,spatial_radius_cells=1.5,normal_limit=60,
                truth_available=False,physical_accuracy_measured=False,production_tracker_changed=False)
    (output/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')


def plot(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for gate in [1.5,2.5]:
        fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
        for col,mode in enumerate(['native_cells','native_cells_scaled_count']):
            for method in ['geometry','translation','expansion']:
                a=[r for r in rows if r['mode']==mode and r['gate_reference_cells']==gate and r['method']==method]
                t=[r['target_time'] for r in a]
                axes[0,col].plot(t,[r['median_group_nearest_distance_cells'] for r in a],'.-',label=method)
                axes[1,col].plot(t,[r['accepted_links'] for r in a],'.-',label=method)
            axes[0,col].set_title(f'{mode.replace("_"," ")}; cutoff {20 if col==0 else 80}')
            axes[0,col].set_ylabel('Median of group median distances (cells)')
            axes[1,col].set(xlabel='Held-out target time',ylabel='Accepted candidate links')
            for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=9)
        fig.suptitle(f'128³ saved-catalog prediction audit | bootstrap and final gate {gate}/64 code units',fontsize=14)
        fig.text(.035,.025,'Distances are to nearest retained future positions, irrespective of normals or physical identity; each group contributes one median.\n'
                 'Link decisions retain signed-normal/pooled-support tests. Lower distances or more links do not establish physical correctness.\n'
                 'Source-overlap samples remain included; excluded groups and initialization failures are recorded. Every triplet is bootstrapped afresh.',fontsize=9)
        fig.tight_layout(rect=(0,.13,1,.94));fig.savefig(output/f'prediction_audit_gate_{str(gate).replace(".","p")}.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args();run(a.directory,a.output_dir)
