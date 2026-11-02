"""NumPy readback of recorded paths; no simulation or JAX execution."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
args=parser.parse_args();root=args.directory
metadata=json.loads((root/'metadata.json').read_text())
assert metadata['status']=='complete'
repo=root.resolve().parents[1]
source_checks={name:hashlib.sha256((repo/name).read_bytes()).hexdigest()==value
               for name,value in metadata['source_sha256'].items()}
assert all(source_checks.values())
np.testing.assert_array_equal(metadata['requested_times'],metadata['actual_times'])
with (root/'endpoint_checks.csv').open() as f: rows=list(csv.DictReader(f))
with np.load(root/'normal_paths.npz') as a:
    assert len(rows)==len(a['grid_indices'])
    s=a['s_cells']
    for side,sign,column in [('upstream',1,0),('downstream',-1,1)]:
        distance=a[f'{side}_distance_cells']
        index=np.argmin(abs(s[None,:]-sign*distance[:,None]),axis=1)
        np.testing.assert_allclose(s[index],sign*distance,rtol=0,atol=1e-6)
        for field in ('pressure','density'):
            interpolated=a[field][np.arange(len(rows)),index]
            np.testing.assert_allclose(interpolated,a['selected_'+field][:,column],rtol=1e-5,atol=1e-7)
        positions=a['surface_positions']+sign*(1/metadata['num_cells'])*distance[:,None]*a['normals']
        radii=np.linalg.norm(positions-.5,axis=1)
        np.testing.assert_allclose(radii,[float(r[f'{side}_radius']) for r in rows],rtol=1e-6,atol=1e-8)
        np.testing.assert_array_equal(a['shock_zone'][np.arange(len(rows)),index]>.5,
                                     [r[f'{side}_in_zone']=='True' for r in rows])
    p_ratio=a['selected_pressure'][:,1]/a['selected_pressure'][:,0]
    np.testing.assert_allclose(np.sqrt((4*p_ratio+1)/5),[float(r['mach']) for r in rows],rtol=1e-5,atol=1e-7)
    np.testing.assert_array_equal(a['grid_indices'],[[int(r[k]) for k in ('ix','iy','iz')] for r in rows])
report=dict(passed=True,directory=str(root),backend=metadata['backend'],samples=len(rows),
    source_hashes_match=source_checks,requested_actual_times_match=True,recorded_path_endpoints_match_selected_values=True,
    saved_endpoint_radii_and_zone_flags_match=True,pressure_mach_relation_matches_by_construction=True,
    full_gas_states_rechecked=False)
(root/'readback_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
