"""Audit matched source budgets before a 64³/128³ morphology comparison."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.wind_bubble.two_bubble_source_budget import budget_plan


def prepare(reference_dir,output,target_cells=128):
    meta_path=reference_dir/'metadata.json';history_path=reference_dir/'online/history.csv'
    reference=json.loads(meta_path.read_text())
    original=budget_plan(reference['num_cells'],reference)
    target_default=budget_plan(target_cells)
    matched=budget_plan(target_cells,reference)
    for key,expected in [('ambient_density',1.),('ambient_pressure',.01),('gamma',5/3)]:
        if not np.isclose(reference[key],expected,rtol=0,atol=1e-12):raise ValueError(f'Reference {key} differs from the runner.')
    if reference['domain']!=[0.,1.]:raise ValueError('Require the unit-cube reference.')
    history=list(csv.DictReader(history_path.open()))
    times=np.array([float(r['time']) for r in history])
    np.testing.assert_allclose(times,reference['requested_times'],atol=1e-6,rtol=0)
    np.testing.assert_allclose(times,np.linspace(0.,reference['t_end'],len(times)),atol=1e-6,rtol=0)
    # Runner metadata can legitimately change with the new CLI option; validate
    # the actual injection/configuration modules against the old experiment.
    physics_hashes={}
    for name in ['astronomix/_modules/_stellar_wind/stellar_wind.py',
                 'astronomix/_modules/_stellar_wind/stellar_wind_options.py',
                 'astronomix/option_classes/simulation_config.py']:
        recorded=next(h for p,h in reference['source_sha256'].items() if p.endswith(name))
        actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual!=recorded:raise ValueError(f'Core source/configuration differs from reference: {name}')
        physics_hashes[name]=actual
    output.mkdir(parents=True,exist_ok=False)
    paths=[meta_path,history_path]
    report=dict(status='ready_for_remote_run',target_run_complete=False,reference_num_cells=reference['num_cells'],
                target_num_cells=target_cells,reference_budget=original,target_unmatched=target_default,target_matched=matched,
                unmatched_energy_increase_percent=100*(target_default['discrete_energy_rates'][0]/original['discrete_energy_rates'][0]-1),
                analysis_events=len(times),t_end=reference['t_end'],requested_times=reference['requested_times'],
                inspection_times=[.04,.08,.12,.13,.14,.16],physics_source_hashes=physics_hashes,
                reference_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                interpretation='Matched discrete total mass/energy rates; injection geometry and numerical source profile still differ. Not proof of convergence.')
    (output/'budget_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    rows=[]
    for mode,plan in [('reference',original),('target_unmatched',target_default),('target_budget_matched',matched)]:
        for source in range(2):
            rows.append(dict(mode=mode,num_cells=plan['num_cells'],source_index=source,
                             injection_cells=plan['injection_cell_counts'][source],mass_parameter=plan['wind_mass_loss_rate_per_source'],
                             parameter_scale=plan['parameter_scale'],discrete_mass_rate=plan['discrete_mass_rates'][source],
                             discrete_energy_rate=plan['discrete_energy_rates'][source]))
    with (output/'source_budgets.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:report[k] for k in ['status','reference_num_cells','target_num_cells','unmatched_energy_increase_percent']},indent=2))
    print('Matched mass-parameter scale:',matched['parameter_scale'])
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference_dir',type=Path);p.add_argument('--target-num-cells',type=int,default=128)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();prepare(a.reference_dir,a.output_dir,a.target_num_cells)
