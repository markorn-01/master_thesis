"""Synchronize measured shock area, thermalization power and accumulated energy.

NumPy-only postprocessing of saved radial-band histories. Missing values remain
missing and cumulative-energy lines never bridge invalid integration intervals.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

KINDS=('forward','reverse')


def read_rows(path):
    with Path(path).open() as f: return list(csv.DictReader(f))


def load_histories(directory):
    directory=Path(directory)
    history=read_rows(directory/'history.csv');energy=read_rows(directory/'energy.csv')
    if not history or len(history)!=len(energy): raise ValueError('Require aligned nonempty history and energy tables.')
    times=np.array([float(r['time']) for r in history])
    if not np.isfinite(times).all() or not (np.diff(times)>0).all(): raise ValueError('Times must be finite and increasing.')
    if not np.allclose(times,[float(r['time']) for r in energy],rtol=0,atol=1e-7):
        raise ValueError('History and energy times differ.')
    if [int(r['event_index']) for r in energy]!=list(range(len(times))): raise ValueError('Energy event numbering differs.')
    quality_path=directory/'sampling_quality.csv'
    quality=read_rows(quality_path) if quality_path.exists() else None
    if quality is not None:
        if len(quality)!=len(times) or not np.allclose(times,[float(r['time']) for r in quality],rtol=0,atol=1e-7):
            raise ValueError('Quality and history event times differ.')
        if [int(r['event_index']) for r in quality]!=list(range(len(times))): raise ValueError('Quality event numbering differs.')
    data={}
    for kind in KINDS:
        power=np.array([float(r[f'{kind}_dissipation_rate']) for r in history])
        area=np.array([float(r[f'{kind}_surface_area']) for r in history])
        cumulative=np.array([float(r[f'{kind}_cumulative_dissipated_energy']) for r in energy])
        raw_flags=[r[f'{kind}_integration_interval_valid'] for r in energy]
        if any(v not in ('True','False') for v in raw_flags): raise ValueError('Invalid integration flag.')
        flags=np.array([v=='True' for v in raw_flags])
        first=np.array([float(r[f'{kind}_first_power_detection_time']) for r in energy])
        duration=np.array([float(r[f'{kind}_integrated_duration']) for r in energy])
        np.testing.assert_allclose(power,[float(r[f'{kind}_dissipation_rate']) for r in energy],rtol=1e-6,atol=1e-8,equal_nan=True)
        # Independent finite-endpoint trapezoids and explicit no-gap-bridging check.
        total=np.nan;origin=np.nan;covered=0.
        for i,rate in enumerate(power):
            valid=False
            if np.isfinite(rate):
                if not np.isfinite(total): total=0.;origin=times[i]
                elif i and np.isfinite(power[i-1]):
                    total+=.5*(power[i-1]+rate)*(times[i]-times[i-1]);covered+=times[i]-times[i-1];valid=True
            np.testing.assert_allclose(cumulative[i],total,rtol=1e-6,atol=1e-8,equal_nan=True)
            np.testing.assert_allclose(first[i],origin,rtol=0,atol=1e-7,equal_nan=True)
            np.testing.assert_allclose(duration[i],covered,rtol=1e-6,atol=1e-8)
            if flags[i]!=valid: raise ValueError('Integration flag differs from finite-endpoint coverage.')
        data[kind]=dict(area=area,power=power,energy=cumulative,valid_interval=flags,
                        first_power_time=first,integrated_duration=duration,
                        energy_point_valid=np.isfinite(power)&np.isfinite(cumulative))
    combined=np.stack([data[k]['energy'] for k in KINDS])
    summed=np.where(np.isfinite(combined).any(axis=0),np.nansum(combined,axis=0),np.nan)
    np.testing.assert_allclose(summed,[float(r['combined_cumulative_dissipated_energy']) for r in energy],rtol=1e-6,atol=1e-8,equal_nan=True)
    return dict(times=times,bands=data,history=history,quality=quality)


def energy_segments(times,values,valid_intervals):
    """Return only explicitly covered adjacent intervals and their end indices."""
    times,values,valid_intervals=map(np.asarray,(times,values,valid_intervals))
    index=np.flatnonzero(valid_intervals & np.isfinite(values))
    index=index[index>0]
    if not np.isfinite(values[index-1]).all(): raise ValueError('Invalid energy interval endpoint.')
    segments=np.stack([np.stack([times[index-1],values[index-1]],axis=1),
                       np.stack([times[index],values[index]],axis=1)],axis=1)
    return segments,index


def render(directory,output_dir,*,fps=3):
    if fps<=0: raise ValueError('FPS must be positive.')
    data=load_histories(directory);times=data['times'];n=len(times)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    colors=dict(forward='#07818c',reverse='#8353b1')
    fig=plt.figure(figsize=(12,9))
    grid=fig.add_gridspec(3,1,left=.12,right=.68,bottom=.16,top=.81,hspace=.22)
    axes=[fig.add_subplot(grid[i]) for i in range(3)]
    labels=['a) Measured shock-surface area','b) Estimated thermalization rate','c) Accumulated thermalized energy']
    units=['Code length²','Code energy / code time','Code energy']
    artists={};energy_collections={};energy_paths={};cursors=[]
    span=times[-1]-times[0] if n>1 else 1.
    for ax,metric,label,unit in zip(axes,('area','power','energy'),labels,units):
        values=np.concatenate([data['bands'][k][metric] for k in KINDS]);finite=values[np.isfinite(values)]
        lower=min(0.,float(finite.min())*1.15) if finite.size else 0.
        upper=max(float(finite.max())*1.18,1e-12) if finite.size else 1.
        ax.set(xlim=(times[0]-.015*span,times[-1]+.015*span),ylim=(lower,upper),ylabel=unit)
        ax.text(.02,.92,label,transform=ax.transAxes,va='top',fontsize=10)
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
        if metric!='energy': ax.tick_params(labelbottom=False)
        for kind in KINDS:
            band=data['bands'][kind];color=colors[kind]
            if metric=='energy':
                segments,index=energy_segments(times,band['energy'],band['valid_interval'])
                ax.add_collection(LineCollection(segments,colors=color,alpha=.18,linewidths=1.4))
                active=LineCollection([],colors=color,linewidths=2);ax.add_collection(active)
                energy_collections[kind]=active;energy_paths[kind]=(segments,index)
                # Faint points also show isolated first measurements without joining gaps.
                mask=band['energy_point_valid'];ax.scatter(times[mask],band['energy'][mask],s=6,color=color,alpha=.18)
            else:
                ax.plot(times,band[metric],color=color,alpha=.18,lw=1.4)
                line,=ax.plot([],[],color=color,lw=2);artists[(kind,metric)]=line
            dot,=ax.plot([],[],'o',color=color,ms=5,zorder=4);artists[(kind,metric,'dot')]=dot
        cursors.append(ax.axvline(times[0],color='#cc6b28',ls='--',lw=1))
    axes[-1].set_xlabel('Simulation time (code units)')
    fig.legend([Line2D([0],[0],color=colors[k],lw=2) for k in KINDS],
               ['Forward radial candidate','Reverse radial candidate'],loc='upper center',bbox_to_anchor=(.43,.9),ncol=2,frameon=False)
    title=fig.suptitle('',fontsize=17,y=.97)
    fig.text(.5,.925,'Measured radial populations • faint curves: complete saved history • solid curves: evolution to current event',ha='center',fontsize=9,color='0.35')
    cards={k:fig.text(.73,y,'',fontsize=10,va='top',linespacing=1.5,color=colors[k],
                     bbox=dict(boxstyle='round,pad=.6',facecolor='#f6f7f8',edgecolor='0.9'))
           for k,y in [('forward',.79),('reverse',.46)]}
    fig.text(.12,.07,'Energy is zero at the first finite power sample, not at simulation start. Missing intervals add no energy and are not bridged.\nThese are estimated shock-heating diagnostics with sampling limits, not the bubble’s total stored energy.\nLabels assume one centered source; no general interacting-shock identities or full-domain coverage are established.',fontsize=8,color='0.35')

    def fmt(value): return f'{value:.4g}' if np.isfinite(value) else 'undefined'
    def update(i):
        for ax,cursor,metric in zip(axes,cursors,('area','power','energy')):
            cursor.set_xdata([times[i],times[i]])
            for kind in KINDS:
                band=data['bands'][kind]
                if metric=='energy':
                    segments,index=energy_paths[kind];energy_collections[kind].set_segments(segments[index<=i])
                else: artists[(kind,metric)].set_data(times[:i+1],band[metric][:i+1])
                valid=band['energy_point_valid'][i] if metric=='energy' else np.isfinite(band[metric][i])
                artists[(kind,metric,'dot')].set_data([times[i]],[band[metric][i]]) if valid else artists[(kind,metric,'dot')].set_data([],[])
        for kind in KINDS:
            b=data['bands'][kind];power=b['power'][i];value=b['energy'][i]
            state=('valid interval' if b['valid_interval'][i] else 'origin / no valid interval' if np.isfinite(power)
                   else 'no power; energy carried' if np.isfinite(value) else 'not yet initialized')
            quality=data['quality']
            overlap=(f'{quality[i][kind+"_upstream_inside_injection"]} / {quality[i][kind+"_surface_samples"]}' if quality is not None else 'not recorded')
            cards[kind].set_text(f'{kind.capitalize()} candidate\nArea: {fmt(b["area"][i])}\nPower: {fmt(power)}\nAccumulated energy: {fmt(value)}\nEnergy origin: t = {fmt(b["first_power_time"][i])}\nCovered duration: {fmt(b["integrated_duration"][i])}\n{state}\nUpstream source overlap: {overlap}')
        title.set_text(f'Shock geometry and heating histories  |  t = {times[i]:.2f}')
        return []

    animation=FuncAnimation(fig,update,frames=n,interval=1000/fps,blit=False,repeat=True)
    animation.save(output_dir/'shock_histories.gif',writer=PillowWriter(fps=fps),dpi=110)
    for i in sorted(set([0,n//2,n-1])):
        update(i);fig.savefig(output_dir/f'preview_{i:04d}.png',dpi=130)
    plt.close(fig)
    from PIL import Image
    durations=[]
    with Image.open(output_dir/'shock_histories.gif') as gif:
        if gif.n_frames!=n: raise AssertionError('GIF misses saved events.')
        for i in range(n): gif.seek(i);gif.load();durations.append(gif.info.get('duration',0))
        size=gif.size
    directory=Path(directory)
    inputs=[directory/'history.csv',directory/'energy.csv']
    if data['quality'] is not None: inputs.append(directory/'sampling_quality.csv')
    metadata=directory.parent/'metadata.json'
    if metadata.exists(): inputs.append(metadata)
    report=dict(passed=True,events=n,decoded_frames=n,times=times.tolist(),fps=fps,frame_size=list(size),frame_durations_ms=durations,
        total_duration_ms=sum(durations),energy_recomputed_from_finite_power_endpoints=True,
        invalid_energy_intervals_not_joined=True,missing_measurements_not_zeroed=True,
        displayed_populations='forward/reverse radial candidates; no combined total curve',
        scope='estimated thermalization since each first finite power; not total stored energy or complete coverage',
        bands={k:dict(first_power_time=float(data['bands'][k]['first_power_time'][-1]),
                      covered_duration=float(data['bands'][k]['integrated_duration'][-1]),
                      final_cumulative_energy=float(data['bands'][k]['energy'][-1])) for k in KINDS},
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output_dir/'render_metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('events','decoded_frames','bands')},indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--fps',type=int,default=3)
    args=parser.parse_args();render(args.directory,args.output_dir,fps=args.fps)

if __name__=='__main__': main()
