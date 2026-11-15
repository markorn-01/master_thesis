"""Case-independent overview of a saved geometric tracking result."""
import argparse
import csv
import json
from pathlib import Path


def plot_overview(graph,counts,mode,path):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(13,9));grid=fig.add_gridspec(2,2,height_ratios=[2,1],left=.07,right=.97,bottom=.16,top=.87,hspace=.35,wspace=.25)
    ax=fig.add_subplot(grid[0,:]);lookup={n['node_id']:n for n in graph['nodes']}
    for event in graph['events']:
        nodes=[lookup[n] for n in event['parents']+event['children']]
        time=np.mean([lookup[event['parents'][0]]['time'],lookup[event['children'][0]]['time']]);level=np.mean([n['segment_id'] for n in nodes])
        color='#b6752c' if event['status']!='shared' else '#167f88' if event['kind']=='continuation' else '#9a4f91'
        for node in nodes:ax.plot([node['time'],time],[node['segment_id'],level],c=color,ls='-' if event['status']=='shared' else '--',lw=1.2)
    ax.scatter([n['time'] for n in graph['nodes']],[n['segment_id'] for n in graph['nodes']],c='#23485e',s=18,zorder=3)
    if graph['segment_count']:
        ax.set_yticks(range(graph['segment_count']));ax.set_ylim(graph['segment_count']-.4,-.6)
    ax.set(ylabel='Local continuation segment',title='Geometric event history');ax.grid(axis='x',alpha=.2)
    time=[float(r['time']) for r in counts];population=fig.add_subplot(grid[1,0]);groups=fig.add_subplot(grid[1,1])
    for key,label in [('detected_samples','All exported samples'),('retained_samples','In retained spatial groups')]:
        population.plot(time,[int(r[key]) for r in counts],label=label)
    population.set(xlabel='Simulation time',ylabel='Surface samples');population.legend(fontsize=9);population.grid(alpha=.2)
    groups.plot(time,[int(r['retained_groups']) for r in counts],'o-')
    groups.set(xlabel='Simulation time',ylabel='Retained geometric groups');groups.grid(alpha=.2)
    fig.suptitle(f'Saved shock-surface history | {mode.replace("_"," ")}',fontsize=16)
    fig.text(.07,.04,'Solid: shared event at both matching distances; dashed orange: distance-specific alternative. Purple: combination/separation.\n'
        'Group/segment counts describe detections, not physical shock birth, death or source ownership. Unlinked observations remain in the graph.\n'
        'No physical ancestry or complete shock coverage is established by this overview.',fontsize=10)
    fig.savefig(path,dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();root=args.directory
    metadata=json.loads((root/'metadata.json').read_text())
    plot_overview(json.loads((root/'graph.json').read_text()),list(csv.DictReader((root/'groups.csv').open())),metadata['mode'],root/'history_overview.png')
