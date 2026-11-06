"""Isolate a floating-weight adapter against frozen unmodified FuzzyCat graphs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import numpy as np
from experiments.astrolink.fuzzycat_float_weights import experimental_float_weight_class
from experiments.astrolink.fuzzycat_inputs import display_memberships,sample_multiplicities
from experiments.astrolink.fuzzycat_weight_reference import reconstruct_memberships


def frozen_clusterer(cls,directory,baseline,n_points):
    """Reuse recorded similarities/order inputs to isolate the weight correction."""
    obj=cls(12,n_points,directoryName=str(directory),minIntraJaccardIndex=.5,
            maxInterJaccardIndex=.5,minStability=.5,windowSize=None,checkpoint=False,verbose=0)
    obj.clusterFileNames=baseline['cluster_files'].copy()
    obj.lazyLoader=[False]*len(obj.clusterFileNames)
    obj.dataTypes=np.zeros(len(obj.clusterFileNames),dtype=np.int8)
    obj._pairs=baseline['pairs'].copy();obj._edges=baseline['pair_jaccard'].copy()
    obj.aggregate()
    for attr,key in [('ordering','ordering'),('groups','groups'),('stabilitiesGroups','group_stabilities'),
                     ('intraJaccardIndicesGroups','group_intra_jaccard'),('interJaccardIndicesGroups','group_inter_jaccard')]:
        np.testing.assert_array_equal(getattr(obj,attr),baseline[key])
    return obj


def plot(output,source,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    affected=[r for r in rows if r['flat_max_absolute_difference']>1e-12]
    if not affected:return
    row=affected[0];name=row['result_directory'];filename=f"result_s{row['min_stability']:g}.npz"
    with np.load(source/f"catalog_{row['event_index']:04d}.npz") as a:
        p,dx=a['positions'],float(a['grid_spacing'])
    with np.load(source/name/filename) as a:old=a['memberships_flat'].max(axis=0)
    with np.load(output/name/filename) as a:new=a['memberships_flat'].max(axis=0)
    section=np.abs(p[:,2]-.5)<=1.5*dx
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,values,title,high,cmap in zip(axes,[old,new,new-old],
                                        ['Unmodified 0.3.2','Experimental floating weights','Change in maximum membership'],
                                        [1,1,max(float(np.abs(new-old).max()),1e-12)],['viridis','viridis','magma']):
        im=ax.scatter(p[section,0],p[section,1],c=values[section],s=12,cmap=cmap,norm=Normalize(0,high))
        fig.colorbar(im,ax=ax,pad=.02,shrink=.8)
        ax.set(title=title,xlim=(.05,.95),ylim=(.15,.85),xlabel='x',ylabel='y',aspect='equal')
    fig.suptitle('Same selected fuzzy group; correcting reciprocal weights changes membership',fontsize=15)
    fig.text(.04,.025,'Affected case: t=0.08, hierarchy input, AstroLink shuffle seed 7, minStability=0.5. Thin section of 3D results.\n'
             'Only reciprocal-weight setup is replaced in an isolated subclass. Jaccard graphs and group selection remain fixed.\n'
             'Membership is an ensemble quantity, not calibrated physical confidence; this comparison does not establish shock identity.',fontsize=9)
    fig.tight_layout(rect=(.01,.16,.99,.9));fig.savefig(output/'membership_correction.png',dpi=140);plt.close(fig)


def run(source,output):
    from fuzzycat import FuzzyCat
    import fuzzycat.fuzzycat as implementation
    cls=experimental_float_weight_class()
    source_meta=json.loads((source/'metadata.json').read_text())
    implementation_hash=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest()
    if implementation_hash!=source_meta['implementation_sha256']:raise RuntimeError('Library source differs from the recorded baseline.')
    output.mkdir(parents=True,exist_ok=False)
    baseline_rows=list(csv.DictReader((source/'summary.csv').open()));rows=[];input_hashes={}
    def record(path):input_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    record(source/'metadata.json');record(source/'summary.csv')
    for name in sorted({r['result_directory'] for r in baseline_rows}):
        target=output/name;target.mkdir()
        bundle_path=source/name/'input_clusters.npz';record(bundle_path)
        with np.load(bundle_path) as a:clusters={key:a[key].copy() for key in a.files}
        selected=[r for r in baseline_rows if r['result_directory']==name]
        n_points=int(selected[0]['n_points'])
        initial_path=source/name/'result_s0.5.npz'
        with tempfile.TemporaryDirectory(prefix='thesis-fuzzycat-float-') as temporary:
            folder=Path(temporary)/'Clusters';folder.mkdir()
            for key,values in clusters.items():np.save(folder/f'{key}.npy',values)
            with np.load(initial_path) as baseline:
                original=frozen_clusterer(FuzzyCat,temporary,baseline,n_points)
                corrected=frozen_clusterer(cls,temporary,baseline,n_points)
            for old_row in selected:
                stability=float(old_row['min_stability']);baseline_path=source/name/f'result_s{stability:g}.npz';record(baseline_path)
                original.minStability=stability;corrected.minStability=stability
                original.extractFuzzyClusters();corrected.extractFuzzyClusters()
                with np.load(baseline_path) as baseline:
                    # Re-executed base library must reproduce the archived output.
                    for attr,key in [('fuzzyClusters','fuzzy_clusters'),('stabilities','stabilities'),
                                     ('memberships','memberships'),('memberships_flat','memberships_flat'),('fuzzyHierarchy','fuzzy_hierarchy')]:
                        np.testing.assert_allclose(getattr(original,attr),baseline[key],rtol=0,atol=1e-12)
                    for attr in ['fuzzyClusters','stabilities','intraJaccardIndices','interJaccardIndices']:
                        np.testing.assert_array_equal(getattr(corrected,attr),getattr(original,attr))
                    counts,weights=sample_multiplicities(corrected.clusterFileNames,corrected.ordering,corrected.fuzzyClusters,12)
                    expected=reconstruct_memberships(corrected.clusterFileNames,clusters,corrected.ordering,corrected.fuzzyClusters,
                                                     weights,corrected.stabilities,12,n_points)
                    for values,reference in zip([corrected.memberships,corrected.memberships_flat,corrected.fuzzyHierarchy],expected):
                        np.testing.assert_allclose(values,reference,rtol=1e-12,atol=1e-12)
                        assert np.isfinite(values).all()
                    flat=corrected.memberships_flat
                    assert (flat>=-1e-12).all() and (flat<=1+1e-12).all() and (flat.sum(axis=0)<=1+1e-12).all()
                    labels,maximum=display_memberships(flat)
                    delta=np.abs(flat-baseline['memberships_flat'])
                    changed=(delta.max(axis=0)>1e-12) if len(flat) else np.zeros(n_points,dtype=bool)
                    row=dict(event_index=int(old_row['event_index']),result_directory=name,min_stability=stability,
                             fuzzy_groups=len(corrected.stabilities),selected_groups_exact=True,baseline_reproduced=True,
                             raw_max_absolute_difference=float(np.abs(corrected.memberships-baseline['memberships']).max()) if flat.size else 0.,
                             flat_max_absolute_difference=float(delta.max()) if flat.size else 0.,changed_samples=int(changed.sum()),
                             old_nonzero_samples=int((baseline['maximum_membership']>0).sum()),new_nonzero_samples=int((maximum>0).sum()),
                             old_display_assigned=int((baseline['display_labels']>=0).sum()),new_display_assigned=int((labels>=0).sum()),
                             display_labels_exact=bool(np.array_equal(labels,baseline['display_labels'])),
                             reference_matches=True,max_flat_membership_sum=float(flat.sum(axis=0).max()))
                    rows.append(row)
                    np.savez_compressed(target/f'result_s{stability:g}.npz',memberships=corrected.memberships,
                                        memberships_flat=flat,fuzzy_hierarchy=corrected.fuzzyHierarchy,
                                        fuzzy_clusters=corrected.fuzzyClusters,stabilities=corrected.stabilities,
                                        ordering=corrected.ordering,cluster_files=corrected.clusterFileNames,
                                        sample_weights=weights,sample_multiplicities=counts,display_labels=labels,maximum_membership=maximum)
        print(f'Compared original and floating-weight execution: {name}',flush=True)
    arguments=(np.arange(4,dtype=np.uint32),np.array([[0,4]],dtype=np.uint32),np.array([0,0,1,1],dtype=np.uint32),2)
    _,old=FuzzyCat._setupHierarchyInformation_njit(*arguments);_,new=cls._setupHierarchyInformation_njit(*arguments)
    np.testing.assert_array_equal(old,[[0],[0]]);np.testing.assert_allclose(new,[[.5],[.5]])
    control=dict(original_weights=old.tolist(),experimental_weights=new.tolist(),expected_weights=[[.5],[.5]],
                 corrected_reciprocal_control_passed=True,base_class_still_unchanged=True)
    (output/'weight_control.json').write_text(json.dumps(control,indent=2)+'\n')
    with (output/'summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in input_hashes.items())
    assert hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest()==implementation_hash
    sources=[Path(__file__),Path('experiments/astrolink/fuzzycat_float_weights.py'),Path('experiments/astrolink/fuzzycat_weight_reference.py')]
    metadata=dict(fuzzycategories=importlib.metadata.version('fuzzycategories'),numpy=np.__version__,numba=importlib.metadata.version('numba'),
                  selections=len(rows),input_sha256=input_hashes,input_hashes_unchanged=True,
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  library_implementation_sha256=implementation_hash,installed_package_unchanged=True,
                  mechanism='Version-scoped subclass overrides only reciprocal-weight setup; fresh aggregation of frozen recorded Jaccard graphs.',
                  scope='Experimental numerical correction, not a production integration or physical shock-identity validation.')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    report=dict(execution_checks_passed=True,selections_compared=len(rows),selected_groups_exact=all(r['selected_groups_exact'] for r in rows),
                all_baselines_reproduced=all(r['baseline_reproduced'] for r in rows),
                floating_reference_matches=all(r['reference_matches'] for r in rows),
                changed_selections=[r for r in rows if r['flat_max_absolute_difference']>1e-12],
                corrected_reciprocal_control_passed=True,installed_package_unchanged=True,physical_validation=False)
    (output/'execution_checks.json').write_text(json.dumps(report,indent=2)+'\n');plot(output,source,rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.directory,a.output_dir)
