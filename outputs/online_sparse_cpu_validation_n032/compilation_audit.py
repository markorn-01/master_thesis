import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from astronomix import time_integration
from experiments.wind_bubble.run_single_bubble import build_problem
from experiments.wind_bubble.run_online_bubble import diagnostic_payload

state, base, params, variables, helper = build_problem(argparse.Namespace(
    num_cells=32, num_snapshots=5, t_end=.2, num_injection_cells=2, plasma_beta=None))
base=base._replace(use_specific_snapshot_timepoints=True, print_elapsed_time=False, progress_bar=False,
                   return_snapshots=True, activate_snapshot_callback=False)
params=params._replace(snapshot_timepoints=jnp.linspace(0.,.2,5),dt_max=.2/5)
snapshots=time_integration(state,base,params,variables)
jax.block_until_ready(snapshots)
analysis=base._replace(return_snapshots=False,activate_snapshot_callback=False)
compiled=jax.jit(lambda physical,h: diagnostic_payload(physical,analysis,variables,h))
results=[]
for i in range(1,5):
    physical=snapshots.states[i]
    eager=diagnostic_payload(physical,analysis,variables,helper)
    fused=compiled(physical,helper)
    jax.block_until_ready(fused)
    mask=np.asarray(eager[0])
    np.testing.assert_array_equal(mask,np.asarray(fused[0]))
    x,y=np.asarray(eager[5])[mask],np.asarray(fused[5])[mask]
    finite=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
    difference=np.abs(x[finite]-y[finite])
    relative=difference/np.abs(x[finite])
    item=dict(event=i,identical_input=True,field='rho_downstream',
              max_absolute_difference=float(difference.max()),
              max_relative_difference=float(relative.max()),
              strict_mismatches=int((~np.isclose(x,y,rtol=1e-6,atol=1e-8,equal_nan=True)).sum()))
    results.append(item)
    print(item,flush=True)
Path('outputs/online_sparse_cpu_validation_n032/compilation_audit.json').write_text(json.dumps(results,indent=2)+'\n')
