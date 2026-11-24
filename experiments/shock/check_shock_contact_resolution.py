"""Observe the same exact shock/contact fan at three grid resolutions."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from experiments.shock.check_colliding_shocks import fingerprint, make_analyzer, write_csv
from experiments.shock.check_shock_contact import evaluate_case
from experiments.shock.colliding_shocks_reference import conserved
from experiments.shock.shock_contact_reference import ShockContactReference


def run(args):
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory.')
    if args.require_gpu and jax.default_backend()!='gpu':
        raise RuntimeError('GPU requested but unavailable.')
    reference=ShockContactReference(contact_density_factor=8.)
    physical_gap=1/64
    sources=[Path(__file__),Path('experiments/shock/shock_contact_reference.py'),
        Path('experiments/shock/colliding_shocks_reference.py'),
        Path('experiments/shock/check_colliding_shocks.py'),Path('experiments/shock/check_shock_contact.py'),
        Path('experiments/wind_bubble/audit_detection_stages.py')]
    sources+=sorted(Path('astronomix/_physics_modules/_shock_finder').glob('*.py'))
    hashes={str(p):fingerprint(p) for p in sources}
    fine,layout=reference.profile(128,physical_gap*128)
    fine_conserved=conserved(fine)
    rows=[]
    (args.output_dir/'profiles').mkdir(parents=True)
    for n in (32,64,128):
        profile,actual=reference.profile(n,physical_gap*n)
        for key in ('contact_x','shock_x','time','riemann_origin'):
            np.testing.assert_allclose(actual[key],layout[key],rtol=0,atol=1e-14)
        ratio=128//n
        np.testing.assert_allclose(conserved(profile),fine_conserved.reshape(n,ratio,3).mean(axis=1),
                                   rtol=1e-13,atol=1e-13)
        analyze=make_analyzer(n)
        row=evaluate_case(analyze,reference,n,'pair',1,0.,physical_gap*n,args.output_dir)
        row['physical_gap']=physical_gap
        rows.append(row)
        print(f'{n}³: same physical gap = {physical_gap*n:g} cells; {row["outcome"]}; Mach={row["mach_median"]}',flush=True)
        del analyze
        jax.clear_caches()
    write_csv(args.output_dir/'summary.csv',rows)
    plot(args.output_dir,rows,reference)
    for path,expected in hashes.items():
        if fingerprint(path)!=expected:
            raise AssertionError(f'Source changed: {path}')
    report=dict(completed=True,backend=jax.default_backend(),jax=jax.__version__,
        physical_gap=physical_gap,physical_time=layout['time'],riemann_origin=layout['riemann_origin'],
        shock_x=layout['shock_x'],contact_x=layout['contact_x'],reference_mach=reference.mach,
        resolutions=[32,64,128],same_physical_solution=True,conservative_coarsening_checked=True,
        filter_cells=1,outcomes=[r['outcome'] for r in rows],mach_medians=[r['mach_median'] for r in rows],
        source_sha256=hashes,production_changed=False,simulation_evolved=False,
        scope='Resolution of one fixed exact shock/contact fan. Not convergence of an evolved bubble simulation.')
    (args.output_dir/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
    return report


def plot(output,rows,reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4.8),dpi=150,sharex=True,sharey=True)
    fig.subplots_adjust(left=.07,right=.98,top=.78,bottom=.26,wspace=.20)
    for ax,row in zip(axes,rows):
        with np.load(output/'profiles'/f'{row["case"]}.npz') as d:
            x=d['x'];contact=float(d['contact_x']);shock=float(d['shock_x'])
            states=np.vstack([reference.states,reference.upstream]);temperature=states[:,2]/states[:,0]
            ax.step([.44,contact,shock,.55],temperature,where='post',ls='--',color='#777777',lw=1.2,label='Exact temperature proxy')
            ax.plot(x,d['temperature'],'o-',color='#087f8c',ms=4,label='Conservative grid input')
            ax.fill_between(x,0,1,where=d['shock_zone'],transform=ax.get_xaxis_transform(),alpha=.14,color='#087f8c')
            for location,color in ((contact,'#713e99'),(shock,'#bb4d34')):
                ax.axvline(location,ls=':',lw=1,color=color)
            ax.set(xlim=(.44,.55),xlabel='x',title=f'{row["resolution"]}³: gap = {row["separation_cells"]:g} cells\n'
                f'Mach = {row["mach_median"]:.3f}' if row['mach_median'] is not None else
                f'{row["resolution"]}³: gap = {row["separation_cells"]:g} cells\nNo detected shock')
            ax.grid(alpha=.2)
    axes[0].set_ylabel('Temperature proxy P/ρ')
    axes[0].legend(fontsize=7)
    fig.suptitle('Same physical Mach-3 shock/contact fan; only grid resolution changes',fontsize=14)
    fig.text(.07,.06,'Physical separation = 1/64. No synthetic broadening. Teal shading: detected shock zone.\n'
        'The finer grid resolves the hot layer between the contact and the shock; this is not an evolved-bubble convergence result.',fontsize=9)
    fig.savefig(output/'fixed_physical_gap.png');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-gpu',action='store_true')
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args())
