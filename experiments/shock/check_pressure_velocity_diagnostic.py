"""Calibrate a read-only pressure/velocity diagnostic on frozen reference inputs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from experiments.shock.check_colliding_shocks import fingerprint,write_csv
from experiments.shock.pressure_velocity_diagnostic import DiagnosticSettings,analyze_state,summarize


def negative_controls(n=64,gamma=5/3):
    """Smooth initial compressions have no entropy jump or physical discontinuity."""
    x=(np.arange(n)+.5)/n
    for width in (1.,2.,4.,8.):
        pressure=1+5*(1-np.tanh((x-.5)*n/width))
        density=pressure**(1/gamma)
        sound=np.sqrt(gamma*pressure/density)
        velocity=2/(gamma-1)*(sound-np.sqrt(gamma))
        yield f'isentropic_compression_w{width:g}',np.stack([density,velocity,pressure])
    # Pressure bump without converging flow and uniform-pressure compression
    # isolate the requirement that the two signals occur together.
    pressure=1+10*np.exp(-((x-.5)/.08)**2)
    yield 'stationary_pressure_bump',np.stack([np.ones(n),np.zeros(n),pressure])
    yield 'uniform_pressure_compression',np.stack([np.ones(n),-x,np.ones(n)])


def run(args):
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory.')
    settings=DiagnosticSettings()
    source_paths=[Path(__file__),Path('experiments/shock/pressure_velocity_diagnostic.py'),
        Path('experiments/wind_bubble/assess_collision_physics.py'),
        Path('astronomix/_physics_modules/_shock_finder/_shock_surface.py')]
    hashes={str(p):fingerprint(p) for p in source_paths}
    inputs={}
    args.output_dir.mkdir(parents=True)
    summaries=[];records=[]
    import csv
    with (args.reference/'summary.csv').open() as stream:
        original={r['case']:r for r in csv.DictReader(stream)}
    inputs[str(args.reference/'summary.csv')]=fingerprint(args.reference/'summary.csv')
    for path in sorted((args.reference/'profiles').glob('*.npz')):
        inputs[str(path)]=fingerprint(path)
        old=original[path.stem]
        with np.load(path) as data:
            state=np.stack([data['density'],data['velocity_x'],data['pressure']])
        n=state.shape[1]
        rows,geometry=analyze_state(state,1/n,settings)
        for row in rows:
            row.update(case=path.stem,input_kind=old['kind'])
        records.extend(rows)
        summary=dict(case=path.stem,input_kind=old['kind'],resolution=n,
            original_outcome=old['outcome'],**summarize(rows,geometry))
        if old['kind']!='contact':
            shock=float(old['shock_x'])
            midpoint=.5*(shock+float(old['contact_x'])) if old['kind']=='pair' else -np.inf
            for label in ('05','10','20'):
                for category in ('pressure_velocity','full_jump'):
                    matches=[r for r in rows if r[f'{category}_{label}pct']
                        and abs(r['x']-shock)*n<=2 and r['x']>midpoint]
                    summary[f'shock_location_{category}_{label}pct']=bool(matches)
        else:
            for label in ('05','10','20'):
                for category in ('pressure_velocity','full_jump'):
                    summary[f'shock_location_{category}_{label}pct']=False
        summaries.append(summary)
    if len(summaries)!=len(original):
        raise AssertionError('Reference cases missing.')
    for name,state in negative_controls():
        rows,geometry=analyze_state(state,1/state.shape[1],settings)
        for row in rows:
            row.update(case=name,input_kind='negative_control')
        records.extend(rows)
        summary=dict(case=name,input_kind='negative_control',resolution=state.shape[1],
            original_outcome='no_shock_expected',**summarize(rows,geometry))
        for label in ('05','10','20'):
            for category in ('pressure_velocity','full_jump'):
                summary[f'shock_location_{category}_{label}pct']=False
        summaries.append(summary)
    write_csv(args.output_dir/'summary.csv',summaries)
    write_csv(args.output_dir/'state_pairs.csv',records)
    make_control_figure(args,settings)
    contacts=[r for r in summaries if r['input_kind']=='contact']
    negatives=[r for r in summaries if r['input_kind']=='negative_control']
    rejected=[r for r in summaries if r['original_outcome']=='no_zone_candidates']
    for path,expected in {**inputs,**hashes}.items():
        if fingerprint(path)!=expected:
            raise AssertionError(f'Input/source changed: {path}')
    report=dict(completed=True,settings=asdict(settings),reference_cases=len(original),
        negative_controls=len(negatives),input_sha256=inputs,source_sha256=hashes,
        diagnostic_only=True,production_changed=False,
        isolated_contacts_flagged_at_10pct=sum(r['pressure_velocity_candidates_10pct'] for r in contacts),
        smooth_compressions_flagged_by_pv_at_10pct=sum(r['pressure_velocity_candidates_10pct'] for r in negatives),
        negative_control_full_jump_flags_at_10pct=sum(r['full_jump_candidates_10pct'] for r in negatives),
        previously_zone_rejected_cases=len(rejected),
        previously_zone_rejected_with_pv_location=sum(r['shock_location_pressure_velocity_10pct'] for r in rejected),
        previously_zone_rejected_with_full_jump_location=sum(r['shock_location_full_jump_10pct'] for r in rejected),
        scope='Candidate and necessary state-jump compatibility diagnostics. PV-only and full-jump compatibility are not proofs of a physical shock; no production result is replaced.',
        references=['https://academic.oup.com/mnras/article/463/1/1026/2589921',
                    'https://arxiv.org/html/1604.07401'])
    (args.output_dir/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('sha256')},indent=2))
    return report


def make_control_figure(args,settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    examples=[]
    for title,name in (
        ('Known isolated shock','n064_shock_c0.25_s0_f1_p0'),
        ('Known unresolved shock + contact','n064_pair_c8_s0.5_f1_p0'),
        ('Known moving contact only','n064_contact_c8_s0_f1_p0'),
    ):
        with np.load(args.reference/'profiles'/f'{name}.npz') as d:
            examples.append((title,np.stack([d['density'],d['velocity_x'],d['pressure']])))
    examples.insert(2,('Known smooth compression',dict(negative_controls())['isentropic_compression_w2']))
    fig,axes=plt.subplots(3,4,figsize=(13,8),dpi=150,sharex=True)
    fig.subplots_adjust(left=.07,right=.98,top=.83,bottom=.18,hspace=.30,wspace=.30)
    for col,(title,state) in enumerate(examples):
        n=state.shape[1];x=(np.arange(n)+.5)/n
        rows,g=analyze_state(state,1/n,settings);summary=summarize(rows,g)
        pv={r['candidate_index'] for r in rows if r['pressure_velocity_10pct']}
        full={r['candidate_index'] for r in rows if r['full_jump_10pct']}
        fields=(state[-1],state[1],state[-1]/state[0]**settings.gamma)
        for ax,values,ylabel in zip(axes[:,col],fields,('Pressure','Velocity','Entropy proxy P/ρᵞ')):
            ax.plot(x,values,'o-',color='#087f8c',lw=1,ms=3)
            for index in pv:
                position=(g['coordinates'][index,0]+.5)/n
                ax.axvline(position,color='#087f8c' if index in full else '#d98a27',ls='--',lw=1.2)
            ax.set(xlim=(.40,.60),ylabel=ylabel)
            ax.grid(alpha=.2)
        axes[0,col].set_title(f'{title}\nPV={summary["pressure_velocity_candidates_10pct"]}, '
                             f'full={summary["full_jump_candidates_10pct"]}',fontsize=10)
        axes[2,col].set_xlabel('x')
    fig.suptitle('A pressure/velocity candidate is not a certified shock',fontsize=15)
    fig.text(.07,.045,'Orange line: pressure/velocity-only candidate. Teal dashed line: also compatible with the full sampled-state checks.\n'
        'The unresolved real shock and the smooth compression both flag PV; density/entropy information prevents automatic promotion.\n'
        'These are separate diagnostic outputs at 10% jump tolerance. Production shock finding and energy statistics are unchanged.',fontsize=9)
    fig.savefig(args.output_dir/'diagnostic_controls.png');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=Path('outputs/shock_contact_reference_n032_n064'))
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args())
