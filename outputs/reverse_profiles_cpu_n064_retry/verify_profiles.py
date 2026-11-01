"""Read back profile inspection products using NumPy only."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
root=Path(__file__).resolve().parent
repo=root.parents[1]
metadata=json.loads((root/'metadata.json').read_text())
assert metadata['status']=='complete'
np.testing.assert_array_equal(metadata['requested_times'],metadata['actual_times'])
source_checks={name:hashlib.sha256((repo/name).read_bytes()).hexdigest()==value
               for name,value in metadata['source_sha256'].items()}
assert all(source_checks.values())

def rows(name):
    with (root/name).open() as f: return list(csv.DictReader(f))

summaries=rows('sample_checks.csv')
report=[]
for index,saved in zip((12,15,20),summaries):
    samples=rows(f'samples_{index:04d}.csv')
    rev=[r for r in samples if r['band']=='reverse']
    valid=[r for r in rev if r['sampling_valid']=='True']
    counts=dict(reverse_samples=len(rev),valid_samples=len(valid),
        compressive_density_samples=sum(float(r['rho2'])>float(r['rho1']) for r in valid),
        pressure_increase_samples=sum(float(r['p2'])>float(r['p1']) for r in valid),
        upstream_inside_injection=sum(float(r['upstream_radius'])<=.0625 for r in valid),
        downstream_inside_injection=sum(float(r['downstream_radius'])<=.0625 for r in valid))
    assert all(int(saved[k])==v for k,v in counts.items())
    assert all(float(r['radius'])>.0625 for r in samples)
    m=np.array([float(r['mach']) for r in valid])
    density_jump=np.array([float(r['rho2'])/float(r['rho1']) for r in valid])
    pressure_jump=np.array([float(r['p2'])/float(r['p1']) for r in valid])
    # gamma=5/3 ideal-gas RH relations; pressure-Mach agreement is by construction.
    np.testing.assert_allclose(m,np.sqrt((4*pressure_jump+1)/5),rtol=1e-5,atol=1e-6)
    expected_density=4*m*m/(m*m+3)
    report.append(dict(time=float(saved['time']),**counts,
        median_density_jump=float(np.median(density_jump)),
        median_rh_density_jump_from_pressure_mach=float(np.median(expected_density)),
        density_rh_fractional_residual_p16_median_p84=np.percentile(density_jump/expected_density-1,[16,50,84]).tolist(),
        upstream_radius_min_median_max=np.percentile([float(r['upstream_radius']) for r in valid],[0,50,100]).tolist()))
result=dict(passed=True,backend=metadata['backend'],source_hashes_match=source_checks,
            requested_times_equal_actual=True,saved_sample_counts_recomputed=True,
            pressure_mach_relation_matches_by_construction=True,events=report,
            full_states_rechecked=False)
(root/'local_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
