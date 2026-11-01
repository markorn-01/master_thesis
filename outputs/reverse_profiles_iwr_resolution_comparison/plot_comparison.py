"""Compare preserved GPU sample distributions; no solver or JAX required."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
rows=[]
for n,color,marker in [(64,'#cc6b28','s'),(128,'#07818c','o')]:
    source=root.parent/f'reverse_profiles_iwr_compgpu3_n{n:03d}'
    data=json.loads((source/'local_checks.json').read_text())['events']
    time=np.array([r['time'] for r in data])
    residual=np.array([r['density_rh_fractional_residual_p16_median_p84'] for r in data])*100
    radii=np.array([r['upstream_radius_min_median_max'] for r in data])
    axes[0].plot(time,residual[:,1],color=color,marker=marker,label=f'{n}³')
    axes[0].fill_between(time,residual[:,0],residual[:,2],color=color,alpha=.15)
    axes[1].plot(time,radii[:,1]-.0625,color=color,marker=marker,label=f'{n}³')
    axes[1].fill_between(time,radii[:,0]-.0625,radii[:,2]-.0625,color=color,alpha=.15)
    for r in data:
        rows.append(dict(num_cells=n,time=r['time'],reverse_samples=r['reverse_samples'],
            upstream_inside_injection=r['upstream_inside_injection'],
            downstream_inside_injection=r['downstream_inside_injection'],
            density_rh_residual_median_percent=100*r['density_rh_fractional_residual_p16_median_p84'][1],
            upstream_min_radius=r['upstream_radius_min_median_max'][0]))
axes[0].set(title='a) Density-jump consistency',ylabel='Median RH fractional residual (%)')
axes[1].set(title='b) Upstream sample clearance',ylabel='Upstream radius − injection radius (code units)')
for ax in axes:
    ax.axhline(0,color='0.4',ls=':',lw=1)
    ax.set_xlabel('Simulation time (code units)')
    ax.set_xticks([.12,.15,.20]);ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False)
fig.suptitle('Reverse-candidate samples: 64³ and 128³ GPU inspections',fontsize=13)
fig.text(.5,.03,'Left shading: 16th–84th sample percentiles. Right shading: sample minimum–maximum.\nThese are population spreads, not confidence intervals. Two resolutions do not establish convergence.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.12,1,.93))
fig.savefig(root/'reverse_sample_resolution_comparison.png',dpi=200)
with (root/'comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
