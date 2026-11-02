"""Recheck source-overlap flags against saved GPU inspection sample radii.

NumPy only: no solver, no reconstruction of missing gas states. This verifies
classification of saved endpoints, not the live distance payload on a GPU.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.wind_bubble.sparse_surfaces import source_overlap_flags


def check_directory(root):
    metadata=json.loads((root/'metadata.json').read_text())
    assert metadata['status']=='complete'
    with (root/'sample_checks.csv').open() as stream:
        expected=list(csv.DictReader(stream))
    assert len(expected)==3
    events=[]
    for index,saved in zip((12,15,20),expected):
        path=root/f'samples_{index:04d}.csv'
        with path.open() as stream:
            samples=[r for r in csv.DictReader(stream) if r['band']=='reverse' and r['sampling_valid']=='True']
        up=np.array([float(r['upstream_radius']) for r in samples])
        down=np.array([float(r['downstream_radius']) for r in samples])
        flags=source_overlap_flags(up,down,np.ones(len(samples),bool),metadata['injection_radius'])
        counts={name:int(flags[name].sum()) for name in ('upstream_inside_injection','downstream_inside_injection')}
        assert flags['sample_positions_valid'].all()
        assert len(samples)==int(saved['valid_samples'])
        for key,value in counts.items():
            assert value==int(saved[key]),(root,index,key)
            np.testing.assert_array_equal(flags[key],[r[key]=='True' for r in samples])
        events.append(dict(time=float(saved['time']),valid_reverse_samples=len(samples),**counts))
    return dict(directory=str(root),num_cells=metadata['num_cells'],backend=metadata['backend'],events=events)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(passed=True,scope='classification from saved endpoint radii; not a live GPU payload test',
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path('experiments/wind_bubble/sparse_surfaces.py'),Path('experiments/wind_bubble/check_sampling_quality.py')]},
                cases=[check_directory(root) for root in args.directories])
    with args.output.open('x') as stream: stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
