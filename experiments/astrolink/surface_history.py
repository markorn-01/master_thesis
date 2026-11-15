"""Reusable offline grouping/history assembly; no source ownership or physics changes."""
from dataclasses import dataclass,asdict
import numpy as np
from experiments.astrolink.shock_connectivity import connectivity_partition
from experiments.astrolink.event_tracking import event_tracking
from experiments.astrolink.expanded_event_tracking import expanded_tracking
from experiments.astrolink.event_history import assemble_history


@dataclass(frozen=True)
class HistorySettings:
    max_partners: int = 3
    grouping_radius_cells: float = 1.5
    min_group_samples: int = 20
    normal_angle: float = 60.
    association_gates: tuple = (1.5/64,2.5/64)

    def __post_init__(self):
        if type(self.max_partners) is not int or self.max_partners not in (3,5):
            raise ValueError('Select baseline max_partners=3 or experimental max_partners=5.')
        if type(self.min_group_samples) is not int or self.min_group_samples<1:
            raise ValueError('Require a positive integer group size.')
        if not np.isfinite(self.grouping_radius_cells) or self.grouping_radius_cells<=0 or not 0<self.normal_angle<90:
            raise ValueError('Invalid spatial grouping parameters.')
        gates=np.asarray(self.association_gates,float)
        if gates.shape!=(2,) or not np.isfinite(gates).all() or not 0<gates[0]<gates[1]:
            raise ValueError('Require two increasing positive physical association gates.')
        if self.normal_angle!=60.:
            raise ValueError('The current temporal model uses the validated 60-degree normal gate.')


def analyze_surface_history(catalogs,times,settings=None,*,progress=None):
    """Group any number of same-grid catalogs and assemble explicitly geometric events.

    Grouping needs only positions/normals/grid_spacing, independent of how many
    sources generated the flow. Event optimization is still a bounded prototype.
    The baseline remains the default; expanded arity is an explicit opt-in.
    """
    settings=HistorySettings() if settings is None else settings
    times=np.asarray(times,float)
    if times.ndim!=1 or len(times)!=len(catalogs) or len(times)<2 or not np.isfinite(times).all() or not np.all(np.diff(times)>0):
        raise ValueError('Require at least two catalogs and matching increasing finite times.')
    partitions=[];dx=None
    for index,catalog in enumerate(catalogs):
        p,n=np.asarray(catalog['positions']),np.asarray(catalog['normals'])
        spacing=float(catalog['grid_spacing'])
        if p.ndim!=2 or p.shape[1:]!=(3,) or n.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(n).all() or (np.linalg.norm(n,axis=1)<=0).any():
            raise ValueError('Require finite (N,3) points and matching nonzero normals.')
        if not np.isfinite(spacing) or spacing<=0 or (dx is not None and dx!=spacing):
            raise ValueError('History requires a fixed positive grid spacing.')
        dx=spacing
        partitions.append(connectivity_partition(p,n,dx,radius_cells=settings.grouping_radius_cells,
            max_angle_degrees=settings.normal_angle,min_display_size=settings.min_group_samples))
        if progress:progress('group',index)
    records=[]
    tracker=event_tracking if settings.max_partners==3 else expanded_tracking
    for index in range(len(catalogs)-1):
        a,b=catalogs[index:index+2]
        for number,gate in enumerate(settings.association_gates):
            result=tracker(a['positions'],a['normals'],partitions[index]['labels'],
                b['positions'],b['normals'],partitions[index+1]['labels'],gate)
            records.append(dict(parent_event=index,child_event=index+1,gate_index=number,
                gate_reference_cells=gate*64,result=result))
        if progress:progress('track',index)
    groups=[list(map(int,np.unique(p['labels'][p['labels']>=0]))) for p in partitions]
    return dict(settings=asdict(settings),partitions=partitions,records=records,
        graph=assemble_history(groups,times,records),experimental=settings.max_partners==5,
        physical_ancestry_established=False)
