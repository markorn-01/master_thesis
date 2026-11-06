"""FuzzyCat stability across same-event AstroLink parameter ensembles."""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import itertools
import json
from pathlib import Path
import tempfile
import platform
import numpy as np
from experiments.astrolink.fuzzycat_inputs import hierarchy_memberships,partition_memberships,display_memberships,sample_multiplicities


def configurations():
    return [f'al_w{w}_k{k}_s{s}' for w,k,s in itertools.product([0,1,4],[20,40],['auto','2'])]


def run_ensemble(archive,event,seed,encoding,include_root,output):
    from fuzzycat import FuzzyCat
    n_points=len(archive['positions']);manifest=[];clusters={}
    name=f'event{event:04d}_o{seed}_{encoding}'+('_with_root' if include_root else '')
    target=output/name;target.mkdir()
    for sample,config in enumerate(configurations()):
        prefix=f'{config}_o{seed}__'
        message=str(archive[prefix+'messages'])
        if 'failed to converge' in message:raise ValueError('Do not include failed AstroLink fits in this declared ensemble.')
        if encoding=='hierarchy':
            groups=hierarchy_memberships(archive[prefix+'ordering_catalog'],archive[prefix+'intervals'],
                                         archive[prefix+'ids'],n_points,include_root)
        else:groups=partition_memberships(archive[prefix+'labels'])
        for identifier,rows in groups.items():clusters[f'{sample:03d}_{identifier}']=rows
        manifest.append(dict(sample=sample,config=config,order_seed=seed,clusters=len(groups),empty=len(groups)==0))
    np.savez_compressed(target/'input_clusters.npz',**clusters)
    (target/'realizations.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summaries=[];messages=io.StringIO()
    # FuzzyCat requires membership .npy files. Keep a compact reproducible bundle
    # in the repository; materialize files only in a temporary working directory.
    with tempfile.TemporaryDirectory(prefix='thesis-fuzzycat-ensemble-') as temporary:
        folder=Path(temporary)/'Clusters';folder.mkdir()
        for key,rows in sorted(clusters.items()):np.save(folder/f'{key}.npy',rows)
        fc=FuzzyCat(12,n_points,directoryName=temporary,minIntraJaccardIndex=.5,
                    maxInterJaccardIndex=.5,minStability=.5,windowSize=None,checkpoint=False,verbose=0)
        with contextlib.redirect_stdout(messages),contextlib.redirect_stderr(messages):
            fc.computeSimilarities()
            pairs,similarities=fc._pairs.copy(),fc._edges.copy()
            fc.aggregate()
            for stability in [.5,.75]:
                fc.minStability=stability;fc.extractFuzzyClusters()
                labels,maximum=display_memberships(fc.memberships_flat)
                memberships=fc.memberships_flat
                multiplicities,reference_weights=sample_multiplicities(fc.clusterFileNames,fc.ordering,fc.fuzzyClusters,12)
                if len(fc.fuzzyClusters):
                    _,library_weights=fc._setupHierarchyInformation_njit(fc.ordering,fc.fuzzyClusters,fc._sampleNumbers,12)
                else:library_weights=np.empty((12,0),dtype=np.int32)
                lost_weights=int(((multiplicities>1)&(library_weights==0)).sum())
                if not np.isfinite(memberships).all():raise AssertionError('Nonfinite FuzzyCat memberships.')
                saved=dict(cluster_files=fc.clusterFileNames,pairs=pairs,pair_jaccard=similarities,
                           ordering=fc.ordering,jaccard_indices=fc.jaccardIndices,
                           groups=fc.groups,group_intra_jaccard=fc.intraJaccardIndicesGroups,
                           group_inter_jaccard=fc.interJaccardIndicesGroups,group_stabilities=fc.stabilitiesGroups,
                           fuzzy_clusters=fc.fuzzyClusters,stabilities=fc.stabilities,
                           intra_jaccard=fc.intraJaccardIndices,inter_jaccard=fc.interJaccardIndices,
                           memberships=fc.memberships,memberships_flat=memberships,fuzzy_hierarchy=fc.fuzzyHierarchy,
                           display_labels=labels,maximum_membership=maximum,min_stability=stability,
                           sample_multiplicities=multiplicities,sample_weights_library=library_weights,
                           sample_weights_float_reference=reference_weights)
                np.savez_compressed(target/f'result_s{stability:g}.npz',**saved)
                summaries.append(dict(event_index=event,time=float(archive['time']),order_seed=seed,encoding=encoding,
                                      include_root=include_root,min_stability=stability,n_points=n_points,n_realizations=12,
                                      empty_realizations=sum(r['empty'] for r in manifest),input_clusters=len(clusters),
                                      fuzzy_groups=len(fc.stabilities),nonzero_membership_samples=int((maximum>0).sum()),
                                      display_assigned_samples=int((labels>=0).sum()),display_tie_samples=int((labels==-3).sum()),
                                      min_membership=float(memberships.min()) if memberships.size else 0.,
                                      max_membership=float(memberships.max()) if memberships.size else 0.,
                                      max_flat_membership_sum=float(memberships.sum(axis=0).max()) if n_points else 0.,
                                      truncated_reciprocal_weight_cases=lost_weights,
                                      result_directory=name))
    (target/'library_messages.txt').write_text(messages.getvalue())
    print(f'{name}: '+', '.join(f"stability {r['min_stability']}: {r['fuzzy_groups']} fuzzy groups" for r in summaries),flush=True)
    return summaries


def plot_sections(output,event=8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    with np.load(output/f'catalog_{event:04d}.npz') as a:p,dx,time=a['positions'],float(a['grid_spacing']),float(a['time'])
    section=np.abs(p[:,2]-.5)<=1.5*dx
    fig,axes=plt.subplots(3,2,figsize=(12,12))
    examples=[('partition','Flat input, root excluded',False),('hierarchy','Hierarchy input, root excluded',False),
              ('hierarchy','Hierarchy input, root included (control)',True)]
    for row,(encoding,title,root) in enumerate(examples):
        name=f'event{event:04d}_o0_{encoding}'+('_with_root' if root else '')
        with np.load(output/name/'result_s0.5.npz') as r:
            labels,maximum=r['display_labels'],r['maximum_membership'];count=len(r['stabilities'])
        colors=np.tile([.65,.65,.65,1.],(len(p),1))
        for j,group in enumerate(np.unique(labels[labels>=0])):colors[labels==group]=plt.get_cmap('tab20')(j%20)
        ax=axes[row,0];ax.scatter(p[section,0],p[section,1],c=colors[section],s=10)
        ax.set_title(f'{title}\n{count} fuzzy groups; {(labels>=0).mean():.1%} assigned for display',fontsize=11)
        im=axes[row,1].scatter(p[section,0],p[section,1],c=maximum[section],cmap='viridis',norm=Normalize(0,1),s=10)
        axes[row,1].set_title('Maximum flattened membership',fontsize=11)
        fig.colorbar(im,ax=axes[row,1],pad=.02,shrink=.8)
        for ax in axes[row]:ax.set(xlim=(.05,.95),ylim=(.15,.85),xlabel='x',ylabel='y',aspect='equal');ax.grid(alpha=.15)
    fig.suptitle(f'Same-event FuzzyCat ensemble at t={time:.2f}',fontsize=16)
    fig.text(.04,.025,'Unmodified FuzzyCat 0.3.2 output: integer reciprocal-weight truncation was detected. Affected membership values need correction/validation.\n'
             'Each ensemble has 12 parameter settings. Display: maximum membership ≥0.5, ties unassigned.\n'
             'Gray: below display threshold or tied. Colors are local, not physical identities. Thin z section displays full-3D memberships.\n'
             'FuzzyCat minStability=0.5; original input order. Membership reflects this declared ensemble, not calibrated physical confidence.',fontsize=9)
    fig.tight_layout(rect=(.01,.14,.99,.94));fig.savefig(output/'contact_memberships.png',dpi=130);plt.close(fig)


def plot_counts(output,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,encoding in zip(axes,['partition','hierarchy']):
        for offset,stability,color in [(-.15,.5,'#007f8b'),(.15,.75,'#bf6c2b')]:
            values=[];low=[];high=[]
            for event in [4,8,16]:
                rows=[r for r in summary if r['event_index']==event and r['encoding']==encoding and not r['include_root'] and r['min_stability']==stability]
                original=next(r['fuzzy_groups'] for r in rows if r['order_seed']==0)
                values.append(original);low.append(original-min(r['fuzzy_groups'] for r in rows));high.append(max(r['fuzzy_groups'] for r in rows)-original)
            ax.bar(np.arange(3)+offset,values,.3,color=color,label=f'minStability {stability:g}',yerr=[low,high],capsize=4)
        ax.set(xticks=np.arange(3),xticklabels=['t=0.04','t=0.08','t=0.16'],ylabel='Selected fuzzy groups',title=f'{encoding.capitalize()} input (no root)')
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('FuzzyCat selection depends on the ensemble and its encoding',fontsize=15)
    fig.text(.04,.02,'Bars: original AstroLink point order. Ranges: min–max across original order and two shuffles; not statistical error bars.\n'
             'Each order has the same 12 parameter choices. No simulations or physical noise realizations are added.',fontsize=9)
    fig.tight_layout(rect=(.01,.14,.99,.91));fig.savefig(output/'fuzzy_group_counts.png',dpi=140);plt.close(fig)


def run(directory,output):
    output.mkdir(parents=True,exist_ok=False);summary=[];hashes={}
    from fuzzycat import FuzzyCat
    _,actual=FuzzyCat._setupHierarchyInformation_njit(np.arange(4,dtype=np.uint32),np.array([[0,4]],dtype=np.uint32),
                                                    np.array([0,0,1,1],dtype=np.uint32),2)
    reproducer=dict(package_version=importlib.metadata.version('fuzzycategories'),
                    description='Two input clusters per sample contribute to one selected fuzzy group.',
                    sample_numbers=[0,0,1,1],expected_reciprocal_weights=[[.5],[.5]],
                    actual_weights=actual.tolist(),actual_dtype=str(actual.dtype),
                    reciprocal_weight_check_passed=bool(np.allclose(actual,[[.5],[.5]])),library_modified=False)
    (output/'weight_reproducer.json').write_text(json.dumps(reproducer,indent=2)+'\n')
    for event in [4,8,16]:
        path=directory/f'event_{event:04d}.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as a:
            np.savez_compressed(output/f'catalog_{event:04d}.npz',positions=a['positions'],normals=a['normals'],time=a['time'],grid_spacing=a['grid_spacing'])
            for seed,encoding in itertools.product([0,42,7],['partition','hierarchy']):
                summary.extend(run_ensemble(a,event,seed,encoding,False,output))
            if event==8:summary.extend(run_ensemble(a,event,0,'hierarchy',True,output))
    with (output/'summary.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    import fuzzycat.fuzzycat as implementation
    source_paths=[Path(__file__),Path('experiments/astrolink/fuzzycat_inputs.py')]
    meta=dict(events=[4,8,16],ensemble_realizations=12,astrolink_parameter_settings=configurations(),order_seeds=[0,42,7],
              encodings=['partition','hierarchy'],primary_root_included=False,root_control='event8, original order, hierarchy',
              min_stabilities=[.5,.75],min_intra_jaccard=.5,max_inter_jaccard=.5,window_size=None,
              n_ensembles=19,n_selections=38,empty_realizations='retained in nSamples denominator',
              python=platform.python_version(),numpy=np.__version__,fuzzycategories=importlib.metadata.version('fuzzycategories'),
              numba=importlib.metadata.version('numba'),implementation_sha256=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),
              input_sha256=hashes,input_hashes_unchanged=True,
              source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
              execution='unmodified FuzzyCat; optional package installed in temporary directory; no JAX/GPU simulation',
              reciprocal_weight_truncation_detected=not reproducer['reciprocal_weight_check_passed'],
              membership_interpretation='Library integer-weight truncation affects some memberships; float references are diagnostics, not corrected memberships.',
              scope='Same-event grouping robustness to declared parameter choices; not temporal tracking or physical identity validation')
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');plot_sections(output);plot_counts(output,summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.directory,a.output_dir)
