"""Compare frozen collision benchmarks before and after opposing-front guards."""
import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from astronomix._physics_modules._shock_finder._shock_zones import get_profile_aware_post_pre_shock_values
from experiments.shock.check_colliding_shocks import fingerprint, write_csv


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def run(args):
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory.')
    old_report=json.loads((args.baseline/'verification.json').read_text())
    new_report=json.loads((args.corrected/'verification.json').read_text())
    for key in ('incoming_mach','outgoing_mach','gamma','widths','separations_cells','cases'):
        if old_report[key]!=new_report[key]:
            raise AssertionError(f'Physical reference or matrix changed: {key}')
    reference='experiments/shock/colliding_shocks_reference.py'
    if old_report['source_sha256'][reference]!=new_report['source_sha256'][reference]:
        raise AssertionError('Independent physical-reference source changed.')
    for path,expected in new_report['source_sha256'].items():
        if fingerprint(path)!=expected:
            raise AssertionError(f'Corrected benchmark source changed: {path}')
    inputs={}
    def record(path):
        inputs[str(path)]=fingerprint(path)
        return path
    for root in (args.baseline,args.corrected):
        record(root/'verification.json')
    old={r['case']:r for r in read_rows(record(args.baseline/'summary.csv'))}
    new={r['case']:r for r in read_rows(record(args.corrected/'summary.csv'))}
    if set(old)!=set(new):
        raise AssertionError('Case matrix differs.')
    comparisons=[]
    for name,before in old.items():
        after=new[name]
        comparisons.append(dict(case=name,resolution=after['resolution'],phase=after['phase'],
            filter_cells=after['filter_cells'],separation_cells=after['separation_cells'],
            before_located=before['all_fronts_located_2cell'],after_located=after['all_fronts_located_2cell'],
            before_accurate=before['all_fronts_located_and_mach_within_5pct'],
            after_accurate=after['all_fronts_located_and_mach_within_5pct'],
            before_mach_error=before['worst_absolute_mach_relative_error'],
            after_mach_error=after['worst_absolute_mach_relative_error']))
    targeted=[]
    for n in (32,64):
        for phase in ('incoming','outgoing'):
            row=new[f'n{n:03d}_{phase}_s2_f1']
            if row['all_fronts_located_and_mach_within_5pct']!='True':
                raise AssertionError(f'Close-front regression remains: {row["case"]}')
            targeted.append(row['case'])
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir/'before_after.csv',comparisons)
    traces=[];figures=[]
    for phase in ('incoming','outgoing'):
        path=record(args.corrected/'profiles'/f'n064_{phase}_s2_f1.npz')
        with np.load(path) as data:
            fields={k:data[k] for k in data.files}
        n=len(fields['x']);direction=jnp.asarray(fields['normal_x'][None],jnp.float32)
        pressure=jnp.asarray(fields['pressure'],jnp.float32)
        density=jnp.asarray(fields['density'],jnp.float32)
        sampled=jax.jit(lambda d,z,p,t,o:get_profile_aware_post_pre_shock_values(
            d,z,p,t,max_steps=8,center_offsets=d*o[None]))(
                direction,jnp.asarray(fields['shock_zone']),pressure,pressure/density,
                jnp.asarray(fields['surface_offsets']))
        p2,p1,_,_,valid,post,pre=map(np.asarray,sampled)
        reconstructed=np.sqrt((4*np.maximum(p2/np.maximum(p1,1e-30),1)+1)/5)
        surface=fields['shock_surface']
        if not valid[surface].all():
            raise AssertionError('Selected close-front samples must be valid.')
        np.testing.assert_allclose(reconstructed[surface],fields['mach'][surface],rtol=1e-6,atol=1e-7)
        case_rows=[]
        for i in np.flatnonzero(surface):
            normal=fields['normal_x'][i]
            position=fields['x'][i]+fields['surface_offsets'][i]*normal/n
            row=dict(phase=phase,cell=int(i),refined_x=float(position),
                pre_x=float(position+pre[i]*normal/n),post_x=float(position-post[i]*normal/n),
                pre_pressure=float(p1[i]),post_pressure=float(p2[i]),
                reference_mach=float(fields['reference_mach']),measured_mach=float(fields['mach'][i]))
            traces.append(row);case_rows.append(row)
        if phase=='incoming':
            np.testing.assert_allclose(p1[surface],1.,atol=1e-6)
            np.testing.assert_allclose(p2[surface],11.,atol=1e-6)
        old_path=record(args.baseline/'profiles'/f'n064_{phase}_s2_f1.npz')
        with np.load(old_path) as data:
            previous={k:data[k] for k in data.files}
        np.testing.assert_array_equal(previous['shock_zone'],fields['shock_zone'])
        figures.append((phase,previous,fields,case_rows))
    write_csv(args.output_dir/'corrected_sampling_trace.csv',traces)
    plot(args.output_dir,figures)
    eligible=[r for r in comparisons if r['phase'] in ('incoming','outgoing')]
    regressions=[r['case'] for r in eligible if r['before_accurate']=='True' and r['after_accurate']!='True']
    improvements=[r['case'] for r in eligible if r['before_accurate']!='True' and r['after_accurate']=='True']
    for path,expected in inputs.items():
        if fingerprint(path)!=expected:
            raise AssertionError(f'Input changed during review: {path}')
    report=dict(completed=True,targeted_cases=targeted,targeted_cases_pass=True,
        before_accurate_cases=sum(r['before_accurate']=='True' for r in eligible),
        after_accurate_cases=sum(r['after_accurate']=='True' for r in eligible),
        improved_cases=improvements,regressed_cases=regressions,
        isolated_and_widest_exact_controls_pass=new_report['isolated_and_widest_exact_mach_within_5pct'],
        contact_surface_samples=new_report['contact_surface_samples'],
        one_dimensional_trace_matches_3d_mach=True,input_sha256=inputs,
        source_sha256={str(Path(__file__)):fingerprint(__file__)},
        scope='Same analytic inputs, corrected detection/sampling. Not bubble validation or hydrodynamic convergence.')
    (args.output_dir/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('sha256')},indent=2))
    return report


def plot(output,figures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),dpi=150)
    fig.subplots_adjust(left=.07,right=.98,top=.80,bottom=.26,wspace=.25)
    for ax,(phase,old,new,rows) in zip(axes,figures):
        x=new['x'];p=new['pressure'];fronts=new['reference_positions']
        ax.plot(x,p,'o-',color='#087f8c',ms=4,label='Pressure from conservative averages')
        for i,front in enumerate(fronts):
            ax.axvline(front,color='#bb4d34',ls='--',lw=1,label='Exact front' if i==0 else None)
        for i,row in enumerate(rows):
            value=np.interp(row['refined_x'],x,p)
            ax.scatter(row['refined_x'],value,marker='v',s=80,color='#713e99',zorder=5,
                       label='Corrected surface' if i==0 else None)
            ax.scatter(row['pre_x'],row['pre_pressure'],marker='s',s=45,color='#e68c18',zorder=6,
                       label='Corrected upstream sample' if i==0 else None)
            ax.annotate('',xy=(row['pre_x'],row['pre_pressure']),xytext=(row['refined_x'],value),
                        arrowprops=dict(arrowstyle='->',color='#e68c18',lw=1.5))
        title=('Incoming: M = 1 → 3 on both fronts' if phase=='incoming'
               else 'Outgoing: no surface → two fronts, M ≈ 1.915')
        ax.set(xlim=(fronts[0]-3/64,fronts[1]+3/64),xlabel='x',ylabel='Pressure',title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='upper center')
    fig.suptitle('Both known two-cell failures corrected | same 64³ input states',fontsize=15)
    fig.text(.07,.055,'Normal-direction guards separate opposing surfaces and keep sampling on the correct side.\n'
        'No shock-zone thresholds or Mach jump relation changed. These close planar cases are separate from the bubble-zone rejection.',fontsize=10)
    fig.savefig(output/'corrected_close_fronts.png');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=Path('outputs/colliding_shocks_reference_n032_n064'))
    parser.add_argument('--corrected',type=Path,default=Path('outputs/colliding_shocks_corrected_n032_n064'))
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args())
