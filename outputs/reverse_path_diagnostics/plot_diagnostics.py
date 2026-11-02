"""Plot stored population statistics; all variations are diagnostics, not errors."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
cases=json.loads((root/'derived_checks.json').read_text())['cases']
with (root/'post_candidate_sensitivity.csv').open() as f: sensitivity=list(csv.DictReader(f))
fig,axes=plt.subplots(1,3,figsize=(13,4.8))
for i,case in enumerate(cases):
    n=case['num_cells'];summary=case['robust_endpoint_variation'];color=('#cc6b28' if n==64 else '#07818c')
    for ax,field in zip(axes[:2],('pressure','density')):
        values=np.array([summary[f'{side}_{field}_one_cell_range_p16_median_p84'] for side in ('upstream','downstream')])*100
        x=np.arange(2)+(i-.5)*.28
        ax.bar(x,values[:,1],width=.26,color=color,label=f'{n}³')
        ax.errorbar(x,values[:,1],yerr=np.stack([values[:,1]-values[:,0],values[:,2]-values[:,1]]),fmt='none',ecolor='0.2',capsize=3)
    records=[r for r in sensitivity if int(r['num_cells'])==n]
    x=[float(r['offset_beyond_downstream_zone_exit_cells']) for r in records]
    y=np.array([[float(r[k]) for k in ('residual_p16','residual_median','residual_p84')] for r in records])*100
    axes[2].plot(x,y[:,1],'o-',color=color,label=f'{n}³')
    axes[2].fill_between(x,y[:,0],y[:,2],color=color,alpha=.15)
for ax,title in zip(axes[:2],('a) Pressure variation near endpoints','b) Density variation near endpoints')):
    ax.set_xticks([0,1],['Upstream','Downstream']);ax.set_ylabel('Range across one cell / central value (%)');ax.set_title(title,fontsize=10)
axes[2].set(title='c) Hypothetical downstream choices',xlabel='Distance beyond zone exit (cells)',ylabel='Median density-RH residual (%)')
axes[2].axhline(0,color='0.4',ls=':',lw=1)
for ax in axes:
    ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False)
fig.suptitle('Reverse-candidate sampling at t=0.20: CPU reruns',fontsize=14)
fig.text(.5,.035,'Bars show medians; whiskers and shading show 16th–84th sample percentiles, not confidence intervals.\nOne cell has different physical widths at the two resolutions. Alternative candidate choices do not change the saved measurements.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.13,1,.92));fig.savefig(root/'endpoint_variation_and_sensitivity.png',dpi=180)
