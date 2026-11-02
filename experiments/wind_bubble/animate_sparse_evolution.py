"""Animate saved sparse shock surfaces and synchronized Mach history; NumPy only.

No solver or JAX initialization. One frame per saved analysis event; no temporal
interpolation or persistent point identities are introduced.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def load_events(directory):
    directory=Path(directory)
    with (directory/'history.csv').open() as stream:
        history=list(csv.DictReader(stream))
    if not history: raise ValueError('History is empty.')
    files=sorted(directory.glob('surfaces_*.npz'))
    if len(files)!=len(history): raise ValueError('Require one sparse catalog for every history row.')
    events=[]
    for i,(row,path) in enumerate(zip(history,files)):
        if path.name!=f'surfaces_{i:04d}.npz': raise ValueError('Catalog numbering must be contiguous from zero.')
        with np.load(path,allow_pickle=False) as archive:
            if int(archive['schema_version']) not in (1,2): raise ValueError('Unsupported catalog schema.')
            positions=archive['positions'].copy();mach=archive['mach'].copy()
            valid=archive['mach_valid'].copy();time=float(archive['time'])
        if positions.shape!=(len(mach),3) or valid.shape!=mach.shape:
            raise ValueError('Inconsistent point/property shapes.')
        if not np.isfinite(positions).all(): raise ValueError('Non-finite surface positions.')
        if not np.isclose(time,float(row['time']),rtol=0,atol=1e-7):
            raise ValueError('Catalog and history times differ.')
        if events and time<=events[-1]['time']: raise ValueError('Event times must increase strictly.')
        if len(mach)!=int(row['surface_cells_outside_injection']):
            raise ValueError('History and sparse populations differ.')
        np.testing.assert_array_equal(valid,np.isfinite(mach)&(mach>0))
        if int(valid.sum())!=int(row['mach_valid_count']): raise ValueError('Valid Mach counts differ.')
        percentiles=np.percentile(mach[valid],[16,50,84]) if valid.any() else [np.nan]*3
        np.testing.assert_allclose(percentiles,[float(row[k]) for k in ('mach_p16','mach_median','mach_p84')],
                                   rtol=1e-6,atol=1e-8,equal_nan=True)
        events.append(dict(time=time,positions=positions,mach=mach,valid=valid,history=row,path=path))
    return events


def render(directory,output_dir,*,fps=3,max_points=100000):
    if fps<=0 or max_points<1: raise ValueError('FPS and max-points must be positive.')
    events=load_events(directory)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    times=np.array([r['time'] for r in events])
    medians=np.array([float(e['history']['mach_median']) for e in events])
    p16=np.array([float(e['history']['mach_p16']) for e in events])
    p84=np.array([float(e['history']['mach_p84']) for e in events])
    values=[e['mach'][e['valid']] for e in events if e['valid'].any()]
    max_mach=float(np.ceil(max(np.max(x) for x in values))) if values else 1.
    nonempty=[e['positions'] for e in events if len(e['positions'])]
    radius=max(float(np.max(np.abs(x-.5))) for x in nonempty)+.04 if nonempty else .5
    bounds=(max(0.,.5-radius),min(1.,.5+radius))
    plt.rcParams.update({'font.size':10})
    fig=plt.figure(figsize=(12,6.4))
    grid=fig.add_gridspec(1,2,width_ratios=[1.2,1],left=.035,right=.96,bottom=.22,top=.84,wspace=.24)
    view=fig.add_subplot(grid[0],projection='3d');history_ax=fig.add_subplot(grid[1])
    cmap=plt.get_cmap('viridis');norm=Normalize(0,max_mach)
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=view,shrink=.62,pad=.03,label='Profile-aware Mach number')
    history_ax.plot(times,medians,color='0.8',lw=1.5,label='Full saved history')
    history_ax.fill_between(times,p16,p84,color='#07818c',alpha=.12,label='16th–84th sample percentiles')
    past,=history_ax.plot([],[],color='#07818c',lw=2,label='Evolution to current event')
    point,=history_ax.plot([],[],'o',color='#cc6b28',ms=7,zorder=4)
    cursor=history_ax.axvline(times[0],color='#cc6b28',ls='--',lw=1)
    history_ax.set(xlabel='Simulation time (code units)',ylabel='Mach number',
                   title='Mach history of detected surface samples',xlim=(times[0]-.015*(times[-1]-times[0]),times[-1]+.015*(times[-1]-times[0])),ylim=(0,max_mach))
    history_ax.spines[['top','right']].set_visible(False);history_ax.grid(alpha=.2)
    history_ax.legend(loc='upper right',fontsize=8,frameon=False)
    stats=fig.text(.59,.075,'',fontsize=10,va='center')
    title=fig.suptitle('',fontsize=17,y=.96)
    fig.text(.5,.895,'Saved shock-surface samples • fixed camera and scales • star: wind source',ha='center',fontsize=10,color='0.35')
    fig.text(.035,.045,'Point cloud, not a connected mesh. Shading shows sample spread, not uncertainty.\nOne frame per analysis event; playback speed is illustrative.',fontsize=8,color='0.4')
    counts=[]
    for e in events: counts.append(min(len(e['mach']),max_points))

    def update(index):
        e=events[index];view.clear()
        selection=np.linspace(0,len(e['mach'])-1,counts[index],dtype=int) if counts[index] else np.array([],dtype=int)
        pos=e['positions'][selection];valid=e['valid'][selection]
        if valid.any():
            view.scatter(*pos[valid].T,c=e['mach'][selection][valid],cmap=cmap,norm=norm,
                         s=5,alpha=.72,linewidths=0,depthshade=False)
        if (~valid).any(): view.scatter(*pos[~valid].T,c='0.7',s=5,alpha=.7,linewidths=0)
        view.scatter([.5],[.5],[.5],marker='*',s=65,c='#cc6b28',edgecolors='white',linewidths=.4)
        if not len(e['mach']):
            view.text2D(.05,.9,'No detected surface samples',transform=view.transAxes,color='0.35')
        view.set(xlim=bounds,ylim=bounds,zlim=bounds,xlabel='x (code units)',ylabel='y (code units)',zlabel='z (code units)',
                 title='3D detected shock surfaces')
        view.set_box_aspect((1,1,1));view.view_init(elev=24,azim=35)
        past.set_data(times[:index+1],medians[:index+1])
        point.set_data([e['time']],[medians[index]]) if np.isfinite(medians[index]) else point.set_data([],[])
        cursor.set_xdata([e['time'],e['time']])
        title.set_text(f'Wind-bubble shock evolution  |  t = {e["time"]:.2f}')
        median=f'{medians[index]:.2f}' if np.isfinite(medians[index]) else 'undefined (no valid samples)'
        stats.set_text(f'Surface samples: {len(e["mach"]):,}    Displayed: {counts[index]:,}\nMedian Mach: {median}')
        return []

    animation=FuncAnimation(fig,update,frames=len(events),interval=1000/fps,blit=False,repeat=True)
    animation.save(output_dir/'shock_evolution.gif',writer=PillowWriter(fps=fps),dpi=110)
    previews=sorted(set([0,len(events)//2,len(events)-1]))
    for i in previews:
        update(i);fig.savefig(output_dir/f'preview_{i:04d}.png',dpi=130)
    plt.close(fig)
    from PIL import Image
    with Image.open(output_dir/'shock_evolution.gif') as gif:
        if gif.n_frames!=len(events): raise AssertionError('GIF does not preserve one frame per event.')
        dimensions=gif.size
    directory=Path(directory)
    input_paths=[directory/'history.csv',*[e['path'] for e in events]]
    metadata=directory.parent/'metadata.json'
    if metadata.exists(): input_paths.append(metadata)
    report=dict(events=len(events),times=times.tolist(),fps=fps,frame_size=list(dimensions),
        rendered_frames=len(events),surface_samples=[len(e['mach']) for e in events],displayed_points=counts,
        fixed_color_limits=[0,max_mach],fixed_spatial_bounds=list(bounds),
        weighting='surface-cell counts; all outside-injection detected samples, not area weighted',
        playback='one frame per saved event; no interpolation; not physical wall-clock speed',
        scene='point clouds of detected shock samples, not fluid particles or persistent point identities',
        validity='Invalid Mach points gray; missing Mach median not assigned zero',
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths},
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output_dir/'render_metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('events','rendered_frames','frame_size','fixed_color_limits')},indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,help='Online output directory containing history.csv and surfaces_*.npz.')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--fps',type=int,default=3)
    parser.add_argument('--max-points',type=int,default=100000)
    args=parser.parse_args();render(args.directory,args.output_dir,fps=args.fps,max_points=args.max_points)

if __name__=='__main__': main()
