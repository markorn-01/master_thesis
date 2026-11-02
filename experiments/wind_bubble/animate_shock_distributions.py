"""Animate sparse shock surfaces with the original saved Mach/density histograms.

Postprocessing only: no JAX, simulation or new shock reconstruction. Histograms
always use all selected surface samples, independent of display subsampling.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.wind_bubble.animate_sparse_evolution import load_events


DISTRIBUTIONS = (('mach','mach','mach_valid'),
                 ('density_jump','density_jump','density_jump_valid'))


def load_distribution_events(directory):
    directory=Path(directory)
    events=load_events(directory)
    files=sorted(directory.glob('histograms_*.npz'))
    if len(files)!=len(events): raise ValueError('Require one histogram archive per saved event.')
    reference_edges={}
    for index,(e,path) in enumerate(zip(events,files)):
        if path.name!=f'histograms_{index:04d}.npz': raise ValueError('Histogram numbering is not contiguous.')
        with np.load(e['path'],allow_pickle=False) as catalog, np.load(path,allow_pickle=False) as hist:
            if not np.isclose(float(hist['time']),e['time'],rtol=0,atol=1e-7):
                raise ValueError('Histogram and surface event times differ.')
            for name,field,flag in DISTRIBUTIONS:
                edges=hist[f'{name}_edges'].copy();counts=hist[f'{name}_counts'].copy()
                if (edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all()
                        or not (np.diff(edges)>0).all()): raise ValueError('Invalid histogram bin edges.')
                if counts.shape!=(len(edges)-1,) or counts.dtype.kind not in 'iu' or (counts<0).any():
                    raise ValueError('Histogram counts must be nonnegative integers matching the bins.')
                if name in reference_edges: np.testing.assert_array_equal(edges,reference_edges[name])
                reference_edges[name]=edges
                values=catalog[field];valid=catalog[flag]
                if values.shape!=(len(e['positions']),) or valid.dtype.kind!='b' or valid.shape!=values.shape:
                    raise ValueError('Histogram properties do not match the selected sample population.')
                np.testing.assert_array_equal(valid,np.isfinite(values)&(values>0))
                accepted=values[valid]
                np.testing.assert_array_equal(np.histogram(accepted,bins=edges)[0],counts)
                info=dict(in_range=int(counts.sum()),invalid=int((~valid).sum()),
                          underflow=int((accepted<edges[0]).sum()),overflow=int((accepted>edges[-1]).sum()))
                row=e['history']
                if int(valid.sum())!=int(row[f'{name}_valid_count']): raise ValueError('Valid counts differ from history.')
                for key in ('invalid','underflow','overflow'):
                    history_key='invalid_count' if key=='invalid' else key
                    if info[key]!=int(row[f'{name}_{history_key}']): raise ValueError('Histogram range accounting differs from history.')
                if sum(info.values())!=len(values): raise ValueError('Histogram accounting loses samples.')
                percentiles=np.percentile(accepted,[16,50,84]) if len(accepted) else [np.nan]*3
                np.testing.assert_allclose(percentiles,[float(row[f'{name}_{key}']) for key in ('p16','median','p84')],
                                           rtol=1e-6,atol=1e-8,equal_nan=True)
                e[name+'_histogram']=dict(edges=edges,counts=counts,median=float(percentiles[1]),**info)
            e['histogram_path']=path
    return events


def render(directory,output_dir,*,fps=3,max_points=100000):
    if fps<=0 or max_points<1: raise ValueError('FPS and max-points must be positive.')
    events=load_distribution_events(directory)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    positive=[e['mach'][e['valid']] for e in events if e['valid'].any()]
    max_mach=float(np.ceil(max(np.max(x) for x in positive))) if positive else 1.
    nonempty=[e['positions'] for e in events if len(e['positions'])]
    radius=max(float(np.max(abs(x-.5))) for x in nonempty)+.04 if nonempty else .5
    bounds=(max(0.,.5-radius),min(1.,.5+radius))
    y_limits={name:max(1,max(int(e[name+'_histogram']['counts'].max()) for e in events))*1.15
              for name,_,_ in DISTRIBUTIONS}
    displayed=[min(len(e['mach']),max_points) for e in events]
    fig=plt.figure(figsize=(13,7.5))
    grid=fig.add_gridspec(2,2,width_ratios=[1.2,1],left=.035,right=.96,bottom=.17,top=.82,
                        wspace=.28,hspace=.85)
    view=fig.add_subplot(grid[:,0],projection='3d')
    axes=[fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,1])]
    norm=Normalize(0,max_mach);cmap=plt.get_cmap('viridis')
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=view,shrink=.66,pad=.025,label='Profile-aware Mach number')
    title=fig.suptitle('',fontsize=17,y=.96)
    fig.text(.5,.9,'All selected samples outside injection • fixed camera, color scale, bins and count axes',ha='center',fontsize=10,color='0.35')
    count_text=fig.text(.045,.115,'',fontsize=10)
    fig.text(.045,.045,'Point cloud; star marks the wind source. Each histogram counts all selected cells, without area weighting.\nOrange line: median of all valid positive values, including values outside the histogram range. Playback speed is illustrative.',fontsize=8,color='0.35')

    def update(index):
        e=events[index];view.clear()
        chosen=np.linspace(0,len(e['mach'])-1,displayed[index],dtype=int) if displayed[index] else np.array([],dtype=int)
        pos=e['positions'][chosen];valid=e['valid'][chosen]
        if valid.any(): view.scatter(*pos[valid].T,c=e['mach'][chosen][valid],cmap=cmap,norm=norm,s=5,alpha=.72,linewidths=0,depthshade=False)
        if (~valid).any(): view.scatter(*pos[~valid].T,c='0.7',s=5,linewidths=0)
        view.scatter([.5],[.5],[.5],marker='*',s=65,c='#cc6b28',edgecolors='white',linewidths=.4)
        if not len(e['mach']): view.text2D(.05,.9,'No detected surface samples',transform=view.transAxes,color='0.35')
        view.set(xlim=bounds,ylim=bounds,zlim=bounds,xlabel='x (code units)',ylabel='y (code units)',zlabel='z (code units)',title='3D detected shock surfaces')
        view.set_box_aspect((1,1,1));view.view_init(elev=24,azim=35)
        for ax,(name,_,_),color,label in zip(axes,DISTRIBUTIONS,('#07818c','#8353b1'),
                                            ('Mach number','Sampled density contrast, ρ₂/ρ₁')):
            data=e[name+'_histogram'];ax.clear()
            ax.stairs(data['counts'],data['edges'],fill=True,color=color,alpha=.8)
            if np.isfinite(data['median']): ax.axvline(data['median'],color='#cc6b28',lw=1.4)
            if name=='density_jump': ax.axvline(1,color='0.35',ls=':',lw=1,label='Ratio = 1');ax.legend(loc='center right',frameon=False,fontsize=8)
            median=f'{data["median"]:.2f}' if np.isfinite(data['median']) else 'undefined'
            ax.set(xlim=(data['edges'][0],data['edges'][-1]),ylim=(0,y_limits[name]),
                   xlabel=label,ylabel='Number of surface cells',title=f'{label.split(",")[0]} distribution  |  median = {median}')
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
            ax.text(.98,.92,f'In range: {data["in_range"]:,}   Invalid: {data["invalid"]:,}\nBelow range: {data["underflow"]:,}   Above range: {data["overflow"]:,}',
                    ha='right',va='top',transform=ax.transAxes,fontsize=8,
                    bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=2))
        title.set_text(f'Wind-bubble shock surfaces and distributions  |  t = {e["time"]:.2f}')
        count_text.set_text(f'Surface samples: {len(e["mach"]):,}    Displayed points: {displayed[index]:,}    Saved event: {index+1}/{len(events)}')
        return []

    animation=FuncAnimation(fig,update,frames=len(events),interval=1000/fps,blit=False,repeat=True)
    animation.save(output_dir/'shock_distributions.gif',writer=PillowWriter(fps=fps),dpi=110)
    for index in sorted(set([0,1,len(events)//2,len(events)-1])):
        update(index);fig.savefig(output_dir/f'preview_{index:04d}.png',dpi=130)
    plt.close(fig)
    from PIL import Image
    duration=[]
    with Image.open(output_dir/'shock_distributions.gif') as gif:
        if gif.n_frames!=len(events): raise AssertionError('GIF does not contain all saved events.')
        for i in range(gif.n_frames): gif.seek(i);gif.load();duration.append(gif.info.get('duration',0))
        size=gif.size
    inputs=[Path(directory)/'history.csv',*[e['path'] for e in events],*[e['histogram_path'] for e in events]]
    metadata=Path(directory).parent/'metadata.json'
    if metadata.exists(): inputs.append(metadata)
    source_paths=[Path(__file__),Path(__file__).with_name('animate_sparse_evolution.py')]
    report=dict(passed=True,events=len(events),times=[e['time'] for e in events],fps=fps,
        decoded_frames=len(events),frame_size=list(size),frame_durations_ms=duration,total_duration_ms=sum(duration),
        saved_histograms_reconstructed_exact=True,history_range_accounting_matches=True,
        original_bins_preserved=True,density_ratios_below_one_retained=True,
        weighting='one count per selected surface cell; not area weighted',
        fixed_spatial_bounds=list(bounds),fixed_mach_color_range=[0,max_mach],fixed_count_axes=y_limits,
        bin_edges={name:events[0][name+'_histogram']['edges'].tolist() for name,_,_ in DISTRIBUTIONS},
        displayed_points=displayed,display_subsampling_affects_histograms=False,
        playback='one frame per saved event; no temporal interpolation or persistent point identities',
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    (output_dir/'render_metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('events','decoded_frames','frame_size','saved_histograms_reconstructed_exact')},indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--fps',type=int,default=3)
    parser.add_argument('--max-points',type=int,default=100000)
    args=parser.parse_args();render(args.directory,args.output_dir,fps=args.fps,max_points=args.max_points)

if __name__=='__main__': main()
