"""Bounded contact-stage sensitivity study on saved, unlabeled shock samples."""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import itertools
import io
import json
from pathlib import Path
import platform
import numpy as np

from experiments.astrolink.check_two_bubble_clustering import feature_arrays, deepest_partition, restore_labels
from experiments.astrolink.shock_connectivity import connectivity_partition

EVENTS = [4,6,7,8,9,10,16]
SEEDS = [0,42,7]


def metrics(labels,eligible):
    selected=labels[eligible]
    ids,counts=np.unique(selected[selected>=0],return_counts=True)
    return dict(display_groups=len(ids),assigned_fraction=float(np.mean(selected>=0)),
                largest_display_group_fraction=float(max(counts,default=0)/len(eligible)))


def plot_evolution(directory,records,stability):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Fixed, declared witness configurations, not selected by desired group count.
    witnesses=[('al_w1_k20_sauto','AstroLink w=1, k=20, S=auto'),
               ('al_w4_k20_sauto','AstroLink w=4, k=20, S=auto'),
               ('graph_r1.5_a30','Graph r=1.5 cells, angle=30 deg'),
               ('graph_r2.5_a60','Graph r=2.5 cells, angle=60 deg')]
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    colors=['#007f8b','#b36b00','#7953b5','#bd3e56']
    for (config,label),color in zip(witnesses,colors):
        times=[];counts=[];coverage=[];agreement=[]
        for index in EVENTS:
            rows=[r for r in records if r['event_index']==index and r['config']==config]
            times.append(rows[0]['time'])
            counts.append([r['display_groups'] for r in rows])
            coverage.append([r['assigned_fraction'] for r in rows])
            agreement.append(min(r['adjusted_rand_all'] for r in stability if r['event_index']==index and r['config']==config))
        times=np.asarray(times);counts=np.asarray(counts);coverage=np.asarray(coverage)
        axes[0].plot(times,counts[:,0],'o-',c=color,label=label)
        axes[0].fill_between(times,counts.min(axis=1),counts.max(axis=1),color=color,alpha=.12)
        axes[1].plot(times,coverage[:,0],'o-',c=color)
        axes[1].fill_between(times,coverage.min(axis=1),coverage.max(axis=1),color=color,alpha=.12)
        axes[2].plot(times,agreement,'o-',c=color)
    axes[0].set(ylabel='Display groups',title='Original order with min–max across the three input orders')
    axes[1].set(ylabel='Assigned fraction',ylim=(-.03,1.03))
    axes[2].set(xlabel='Simulation time',ylabel='Minimum order agreement (ARI)',ylim=(-.05,1.05))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.legend(loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    fig.suptitle('Contact-stage grouping: fixed witness settings',fontsize=17,y=.99)
    fig.text(.07,.025,'ARI ignores label renumbering. High agreement with zero coverage is trivial. Graph counts omit components <20 points for display only.\nLines connect inspected events; they do not establish temporal correspondence or physical shock identities.',fontsize=9)
    fig.tight_layout(rect=(.02,.08,.99,.87))
    fig.savefig(directory/'contact_evolution.png',dpi=130);plt.close(fig)


def plot_sections(directory,index):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    configs=[('al_w1_k20_sauto','AstroLink w=1'),('al_w4_k20_sauto','AstroLink w=4'),
             ('graph_r1.5_a30','Graph r=1.5, angle=30'),('graph_r2.5_a60','Graph r=2.5, angle=60')]
    with np.load(directory/f'event_{index:04d}.npz') as a:
        p,dx,sources,time=a['positions'],float(a['grid_spacing']),a['sources'],float(a['time'])
        keep=np.abs(p[:,2]-.5)<=1.5*dx
        fig,axes=plt.subplots(2,4,figsize=(16,9))
        for row,seed in enumerate([0,42]):
            for column,(config,title) in enumerate(configs):
                labels=a[f'{config}_o{seed}__labels'];groups=np.unique(labels[labels>=0])
                colors=np.tile([.65,.65,.65,.7],(len(p),1))
                for k,group in enumerate(groups):colors[labels==group]=plt.get_cmap('turbo')((k+.5)/max(1,len(groups)))
                ax=axes[row,column]
                ax.scatter(p[keep,0],p[keep,1],c=colors[keep],s=11,edgecolors='none')
                ax.scatter(sources[:,0],sources[:,1],marker='*',s=70,c='black')
                ax.set(xlim=(.05,.95),ylim=(.15,.85),xlabel='x',ylabel='y')
                ax.set_title(f'{title}\n{len(groups)} groups, {np.mean(labels>=0):.1%} assigned',fontsize=11)
                ax.set_aspect('equal');ax.grid(alpha=.1)
                ax.text(.03,.04,'Original order' if seed==0 else 'Shuffled seed 42',transform=ax.transAxes,fontsize=9)
        fig.suptitle(f'Contact-stage comparison at t={time:.2f}',fontsize=17)
        fig.text(.04,.035,'Thin z section for display only. All eligible 3D points entered each method. Colors are local labels.\nGray: no non-root AstroLink assignment or graph component <20 points. No source labels enter either algorithm.',fontsize=10)
        fig.subplots_adjust(left=.05,right=.98,bottom=.14,top=.86,hspace=.55,wspace=.3)
        fig.savefig(directory/f'event_{index:04d}_sections.png',dpi=130);plt.close(fig)


def run(args):
    from astrolink import AstroLink
    from sklearn.metrics import adjusted_rand_score
    args.output_dir.mkdir(parents=True,exist_ok=False)
    records=[];stability=[];input_hashes={}
    for event in EVENTS:
        path=args.directory/f'surfaces_{event:04d}.npz'
        input_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as a:
            if int(a['schema_version'])!=3:raise ValueError('Require multi-source schema 3.')
            p,n,sources=a['positions'],a['normals'],a['source_positions']
            dx,t=float(a['grid_spacing']),float(a['time'])
        _,eligible=feature_arrays(p,n,dx,0.)
        saved=dict(positions=p,normals=n,sources=sources,time=t,grid_spacing=dx,eligible_rows=eligible)
        labels_by_config={}

        def record(config,seed,algorithm,labels,extra):
            row=dict(event_index=event,time=t,algorithm=algorithm,config=config,order_seed=seed,
                     samples=len(p),eligible_samples=len(eligible),**metrics(labels,eligible),
                     normal_weight='',k_den='',S_request='',S_effective='',radius_cells='',angle_degrees='',
                     nonroot_hierarchy_entries='',raw_graph_components='',edges='',
                     model_fit_warning=False,library_messages='')
            row.update(extra);records.append(row)
            labels_by_config.setdefault(config,[]).append(labels[eligible])
            saved[f'{config}_o{seed}__labels']=labels

        for weight,k_den,sig in itertools.product([0.,1.,4.],[20,40],['auto',2.]):
            features,indices=feature_arrays(p,n,dx,weight)
            np.testing.assert_array_equal(indices,eligible)
            config=f'al_w{weight:g}_k{k_den}_s{sig if isinstance(sig,str) else f"{sig:g}"}'
            for seed in SEEDS:
                perm=np.arange(len(features)) if seed==0 else np.random.default_rng(seed).permutation(len(features))
                c=AstroLink(features[perm],d_intrinsic=2,adaptive=0,k_den=k_den,S=sig,
                            h_style=1,workers=1,working_memory=256*1024**2)
                messages=io.StringIO()
                with contextlib.redirect_stdout(messages),contextlib.redirect_stderr(messages):
                    c.run()
                diagnostics=messages.getvalue().strip()
                labels=restore_labels(deepest_partition(c.ordering,c.clusters,c.ids,len(features)),perm,indices,len(p))
                record(config,seed,'astrolink',labels,dict(normal_weight=weight,k_den=k_den,S_request=sig,
                       S_effective=float(c.S),nonroot_hierarchy_entries=len(c.ids)-1,
                       model_fit_warning='failed to converge' in diagnostics,
                       library_messages=diagnostics))
                prefix=f'{config}_o{seed}__'
                density=np.full(len(p),np.nan);density[indices[perm]]=c.logRho
                saved.update({prefix+'ordering_catalog':indices[perm[c.ordering]],
                              prefix+'intervals':c.clusters,prefix+'ids':c.ids,
                              prefix+'significances':c.significances,prefix+'density_catalog':density,
                              prefix+'fit_shape':c.pFit,prefix+'messages':np.asarray(diagnostics)})
        for radius,angle in itertools.product([1.5,2.5],[30.,60.]):
            config=f'graph_r{radius:g}_a{angle:g}'
            for seed in SEEDS:
                perm=np.arange(len(p)) if seed==0 else np.random.default_rng(seed).permutation(len(p))
                result=connectivity_partition(p[perm],n[perm],dx,radius,angle,20)
                labels=np.empty(len(p),dtype=np.int32);labels[perm]=result['labels']
                raw=np.empty(len(p),dtype=np.int32);raw[perm]=result['raw_labels']
                record(config,seed,'normal_graph',labels,dict(radius_cells=radius,angle_degrees=angle,
                       raw_graph_components=len(result['component_sizes']),edges=len(result['edges'])))
                saved[f'{config}_o{seed}__raw_labels']=raw
        for config,labels in labels_by_config.items():
            for i,j in itertools.combinations(range(len(SEEDS)),2):
                joint=(labels[i]>=0)&(labels[j]>=0)
                stability.append(dict(event_index=event,time=t,config=config,order_a=SEEDS[i],order_b=SEEDS[j],
                                      adjusted_rand_all=float(adjusted_rand_score(labels[i],labels[j])),
                                      jointly_assigned_fraction=float(np.mean(joint)),
                                      adjusted_rand_joint=float(adjusted_rand_score(labels[i][joint],labels[j][joint])) if joint.sum()>=2 else np.nan))
        np.savez_compressed(args.output_dir/f'event_{event:04d}.npz',**saved)
        print(f'event {event:04d}, t={t:.2f}: completed 36 AstroLink and 12 graph runs',flush=True)
    for name,rows in [('summary',records),('order_stability',stability)]:
        with (args.output_dir/f'{name}.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if not all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in input_hashes.items()):
        raise AssertionError('Input catalogs changed during analysis.')
    metadata=dict(status='complete',events=EVENTS,order_seeds=SEEDS,astrolink_runs=252,graph_runs=84,
                  python=platform.python_version(),astrolink=importlib.metadata.version('astrolink'),
                  numpy=np.__version__,scipy=importlib.metadata.version('scipy'),
                  al_normal_weights=[0,1,4],al_k_den=[20,40],al_S=['auto',2.],
                  adaptive=0,intrinsic_dimension=2,h_style=1,workers=1,
                  graph_radius_cells=[1.5,2.5],graph_max_angle_degrees=[30,60],graph_min_display_size=20,
                  graph_rule='undirected proximity edges with signed normal compatibility; no curvature or tangent-plane check',
                  excluded_features='no source positions or ownership, Mach, density or time in either clustering metric',
                  input_sha256=input_hashes,input_hashes_unchanged=True,library_messages_captured=True,
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),
                    Path('experiments/astrolink/shock_connectivity.py'),Path('experiments/astrolink/check_two_bubble_clustering.py')]},
                  interpretation='local surface segmentation candidates, not physical or temporal identity validation')
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    plot_evolution(args.output_dir,records,stability)
    for event in [4,8,16]:plot_sections(args.output_dir,event)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
