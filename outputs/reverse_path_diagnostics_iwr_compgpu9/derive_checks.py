"""Recompute endpoint-gradient and candidate-choice diagnostics from saved paths."""
from pathlib import Path
import csv
import json
import numpy as np
root=Path(__file__).resolve().parent
results=[];sensitivity=[]
for n in (64,128):
    case=root.parent/f'reverse_paths_iwr_compgpu9_n{n:03d}'
    with (case/'endpoint_checks.csv').open() as f: rows=list(csv.DictReader(f))
    summary=json.loads((case/'summary.json').read_text())
    with np.load(case/'normal_paths.npz') as a:
        p1=a['selected_pressure'][:,0,None];rho1=a['selected_density'][:,0,None]
        p2=a['candidate_pressure'][:,6:];rho2=a['candidate_density'][:,6:]
        m2=(4*p2/p1+1)/5
        expected=4*m2/(m2+3)
        residual=(rho2/rho1)/expected-1
        valid=np.isfinite(residual)&(p1>0)&(rho1>0)&(p2>=p1)&(rho2>0)
        residual=np.where(valid,residual,np.nan)
        for j,offset in enumerate(np.arange(.25,1.501,.25)):
            sensitivity.append(dict(num_cells=n,offset_beyond_downstream_zone_exit_cells=float(offset),
                valid_pairs=int(valid[:,j].sum()),
                residual_p16=float(np.nanpercentile(residual[:,j],16)),
                residual_median=float(np.nanpercentile(residual[:,j],50)),
                residual_p84=float(np.nanpercentile(residual[:,j],84))))
        robust = {}
        neighborhoods = {}
        s=a['s_cells']
        for side,sign in [('upstream',1),('downstream',-1)]:
            endpoints=sign*a[f'{side}_distance_cells']
            targets=endpoints[:,None]+np.array([-.5,0,.5])
            indices=np.argmin(abs(s[None,None,:]-targets[:,:,None]),axis=-1)
            np.testing.assert_allclose(s[indices],targets,rtol=0,atol=1e-6)
            neighborhoods[side] = dict(
                samples_with_zone_in_one_cell_interval=int((np.take_along_axis(a['shock_zone'],indices,axis=1)>.5).any(axis=1).sum()),
                samples_with_injection_in_one_cell_interval=int((np.take_along_axis(a['path_radius'],indices,axis=1)<=.0625).any(axis=1).sum()))
            for field in ('pressure','density'):

                values=np.take_along_axis(a[field],indices,axis=1)
                variation=(np.max(values,axis=1)-np.min(values,axis=1))/np.maximum(abs(values[:,1]),1e-30)
                robust[f'{side}_{field}_one_cell_range_p16_median_p84']=np.percentile(variation,[16,50,84]).tolist()
        selected_post=np.array([int(r['downstream_candidate_index']) for r in rows])
        np.testing.assert_allclose(residual[np.arange(len(rows)),selected_post],
            [float(r['rh_density_residual']) for r in rows],rtol=2e-4,atol=1e-6)
        up=np.array([float(r['upstream_radius']) for r in rows])
        down=np.array([float(r['downstream_radius']) for r in rows])
        assert (up>.0625).all() and (down>.0625).all()
        results.append(dict(num_cells=n,reverse_samples=len(rows),summary=summary,robust_endpoint_variation=robust,endpoint_neighborhoods=neighborhoods))
with (root/'post_candidate_sensitivity.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(sensitivity[0]));writer.writeheader();writer.writerows(sensitivity)
(root/'derived_checks.json').write_text(json.dumps(dict(passed=True,cases=results,
    sensitivity_scope='hypothetical downstream choices among the six existing candidates; same selected upstream, no estimator changes',
    note='GPU reruns on compgpu9, physical GPU 3; same-estimator path inspection'),indent=2)+'\n')
print(json.dumps(sensitivity,indent=2))
