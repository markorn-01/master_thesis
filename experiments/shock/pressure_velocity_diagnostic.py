"""Read-only pressure/convergence candidates with separate state-jump checks.

This is an experimental diagnostic, not a replacement shock finder. Pressure
and velocity can flag a front without establishing two resolved shock states.
No candidate is added to production surfaces, statistics, or heating outputs.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import map_coordinates

from astronomix._physics_modules._shock_finder._shock_surface import (
    _find_shock_surface_3d, _refine_surface_offsets_3d,
)
from experiments.wind_bubble.assess_collision_physics import pair_metrics


@dataclass(frozen=True)
class DiagnosticSettings:
    gamma: float = 5/3
    minimum_mach: float = 1.3
    half_widths: tuple = (.5,1.5,2.5,3.5)
    tolerances: tuple = (.05,.10,.20)
    entropy_signal_tolerance: float = .5
    endpoint_step: float = .5


def jump_diagnostics(upstream, downstream, upstream_to_downstream, gamma=5/3):
    """Evaluate one pair without interpreting compatibility as proof of a shock.

    Pressure predicts a Mach number, density ratio, normal velocity jump, and
    entropy increase. Compare those independent predictions to sampled states.
    The mass-inferred speed in ``pair_metrics`` is not independent evidence.
    """
    up,down,n=map(lambda q:np.asarray(q,float),(upstream,downstream,upstream_to_downstream))
    metrics=pair_metrics(up,down,n,gamma)
    ratio=down[4]/up[4]
    mach=np.sqrt(((gamma+1)*ratio+(gamma-1))/(2*gamma))
    compression=((gamma+1)*ratio+(gamma-1))/((gamma-1)*ratio+(gamma+1))
    c1=np.sqrt(gamma*up[4]/up[0])
    expected_du=2*c1/(gamma+1)*(mach-1/mach)
    actual_du=float((up[1:4]-down[1:4])@n)
    log_entropy=np.log(ratio)-gamma*np.log(compression)
    measured_log_entropy=np.log(ratio)-gamma*np.log(down[0]/up[0])
    # These comparisons are diagnostic only. Very weak shocks have tiny
    # entropy signals, so the separate 1.3 Mach floor is retained.
    metrics.update(pressure_mach=float(mach),pressure_predicted_density_ratio=float(compression),
        pressure_predicted_velocity_drop=float(expected_du),
        pressure_velocity_relative_error=float(abs(actual_du-expected_du)/max(abs(expected_du),1e-12*c1)),
        density_jump_relative_error=float(abs((down[0]/up[0])/compression-1)),
        predicted_log_entropy=float(log_entropy),
        entropy_signal_relative_error=float(abs(measured_log_entropy-log_entropy)/max(abs(log_entropy),1e-12)))
    return metrics


def candidate_geometry(state, grid_spacing, guard_distance=4.):
    """Find local compression peaks along pressure normals, with no T criterion.

    Canonical state shape is (ndim+2, *spatial_shape): rho, velocity components,
    pressure. A nonzero pressure gradient and negative divergence only propose
    a location; pressure, velocity, and state consistency are checked afterward.
    """
    q=np.asarray(state,float)
    ndim=q.ndim-1
    if ndim not in (1,2,3) or q.shape[0]!=ndim+2 or min(q.shape[1:])<6:
        raise ValueError('Expected rho, ndim velocity components, pressure on a grid of at least 6 cells.')
    if not np.isfinite(q).all() or min(float(q[0].min()),float(q[-1].min()))<=0:
        raise ValueError('Require finite positive density and pressure.')
    if not np.isfinite(grid_spacing) or grid_spacing<=0:
        raise ValueError('Grid spacing must be finite and positive.')
    pressure=q[-1]
    gradient=np.stack([np.gradient(pressure,grid_spacing,axis=axis) for axis in range(ndim)])
    magnitude=np.sqrt((gradient*gradient).sum(axis=0))
    divergence=sum(np.gradient(q[1+axis],grid_spacing,axis=axis) for axis in range(ndim))
    gradient_floor=1e-10*max(1.,float(pressure.max()))/grid_spacing
    hot_to_cold=np.where((magnitude>gradient_floor)[None],-gradient/np.maximum(magnitude[None],1e-30),0.)
    zone=(magnitude>gradient_floor)&(divergence<0)
    interior=np.zeros(pressure.shape,bool)
    interior[(slice(2,-2),)*ndim]=True
    zone&=interior
    # These geometry helpers follow arbitrary-length normals; their scalar
    # interpolation routines also support the 1D reference profiles.
    surface=np.asarray(_find_shock_surface_3d(jnp.asarray(divergence),jnp.asarray(zone),jnp.asarray(hot_to_cold)))
    offsets=np.asarray(_refine_surface_offsets_3d(jnp.asarray(divergence),jnp.asarray(surface),jnp.asarray(hot_to_cold)))
    indices=np.argwhere(surface)
    normals=hot_to_cold[(slice(None),)+tuple(indices.T)].T
    coordinates=indices+offsets[tuple(indices.T)][:,None]*normals
    # Fixed-width diagnostics must not silently cross an opposing front.
    # Reject such sampling windows instead of inventing plateau samples by
    # repeating the final point at a cap.
    limits=[]
    for sign in (-1.,1.): # Relative to the hot-to-cold direction.
        limit=np.full(len(indices),np.inf)
        for distance in np.arange(.25,guard_distance+.125,.25):
            sample=coordinates+sign*distance*normals
            current=np.stack([map_coordinates(component,sample.T,order=0,mode='nearest')
                              for component in hot_to_cold],axis=1)
            opposite=(current*normals).sum(axis=1)<0
            limit=np.where(opposite&~np.isfinite(limit),distance-.25,limit)
        limits.append(limit)
    return dict(indices=indices,coordinates=coordinates,normals=normals,
                divergence=divergence[tuple(indices.T)],candidate_zone=zone,surface=surface,
                downstream_limit=limits[0],upstream_limit=limits[1])


def analyze_state(state, grid_spacing, settings=DiagnosticSettings(), sources=None, source_radius=.0625,
                  candidate_bounds=None):
    """Return sampled-pair records and candidate locations without writing files."""
    q=np.asarray(state,float)
    ndim=q.ndim-1
    if any(w<=0 or not np.isfinite(w) for w in settings.half_widths):
        raise ValueError('Half widths must be finite and positive.')
    if settings.minimum_mach<=1 or settings.endpoint_step<=0 or not 0<settings.entropy_signal_tolerance:
        raise ValueError('Require a supersonic floor and positive diagnostic limits.')
    if sources is not None and ndim!=3:
        raise ValueError('Source exclusion is defined for the 3D bubble inputs.')
    geometry=candidate_geometry(q,grid_spacing,max(settings.half_widths)+settings.endpoint_step)
    if candidate_bounds is not None:
        bounds=np.asarray(candidate_bounds,float)
        if bounds.shape!=(ndim,2) or not np.all(bounds[:,0]<bounds[:,1]):
            raise ValueError('Bounds must have one increasing interval per dimension.')
        positions=(geometry['coordinates']+.5)*grid_spacing
        selected=((positions>=bounds[:,0])&(positions<=bounds[:,1])).all(axis=1)
        for key in ('indices','coordinates','normals','divergence','upstream_limit','downstream_limit'):
            geometry[key]=geometry[key][selected]
    centers=geometry['coordinates'];normal=-geometry['normals'] # Upstream toward downstream.
    def sample(coordinates):
        return np.stack([map_coordinates(field,coordinates.T,order=1,mode='nearest') for field in q],axis=1)
    def primitive_3d(values):
        output=np.zeros((len(values),5));output[:,0]=values[:,0];output[:,4]=values[:,-1]
        output[:,1:1+ndim]=values[:,1:1+ndim]
        return output
    bounds=np.asarray(q.shape[1:])-1
    rows=[]
    geometry['sampling_windows']=[]
    for half_width in settings.half_widths:
        up_points=centers-half_width*normal
        down_points=centers+half_width*normal
        extra=settings.endpoint_step
        up_outer=centers-(half_width+extra)*normal
        down_outer=centers+(half_width+extra)*normal
        in_bounds=np.logical_and.reduce([(points>=0).all(axis=1)&(points<=bounds).all(axis=1)
                                         for points in (up_points,down_points,up_outer,down_outer)])
        out_of_bounds=int((~in_bounds).sum())
        source_intersections=0
        if sources is not None:
            from experiments.wind_bubble.assess_collision_physics import source_free_segments
            a=(up_outer+.5)*grid_spacing;b=(down_outer+.5)*grid_spacing
            source_free=source_free_segments(a,b,np.asarray(sources),source_radius)
            source_intersections=int((in_bounds&~source_free).sum())
            in_bounds&=source_free
        geometry['sampling_windows'].append(dict(half_width_cells=half_width,
            geometric_candidates=len(centers),out_of_bounds=out_of_bounds,
            source_intersections=source_intersections,sampled_pairs=int(in_bounds.sum())))
        upstream,downstream=primitive_3d(sample(up_points)),primitive_3d(sample(down_points))
        outer_up,outer_down=primitive_3d(sample(up_outer)),primitive_3d(sample(down_outer))
        padded_normal=np.zeros((len(normal),3));padded_normal[:,:ndim]=normal
        for i in np.flatnonzero(in_bounds):
            a,b=upstream[i],downstream[i];n=padded_normal[i]
            metrics=jump_diagnostics(a,b,n,settings.gamma)
            dp=abs(b[4]-a[4]);du=abs(float((a[1:4]-b[1:4])@n))
            c1=np.sqrt(settings.gamma*a[4]/a[0])
            pressure_scale=max(dp,1e-12*max(a[4],b[4]))
            velocity_scale=max(du,1e-12*c1)
            plateau=max(abs(outer_up[i,4]-a[4])/pressure_scale,
                        abs(outer_down[i,4]-b[4])/pressure_scale,
                        abs(float((outer_up[i,1:4]-a[1:4])@n))/velocity_scale,
                        abs(float((outer_down[i,1:4]-b[1:4])@n))/velocity_scale)
            row=dict(candidate_index=int(i),half_width_cells=half_width,
                x=float((centers[i,0]+.5)*grid_spacing),
                y=float((centers[i,1]+.5)*grid_spacing) if ndim>1 else 0.,
                z=float((centers[i,2]+.5)*grid_spacing) if ndim>2 else 0.,
                normal_x=float(n[0]),normal_y=float(n[1]),normal_z=float(n[2]),
                velocity_divergence=float(geometry['divergence'][i]),
                upstream_density=float(a[0]),downstream_density=float(b[0]),
                upstream_pressure=float(a[4]),downstream_pressure=float(b[4]),
                endpoint_pressure_velocity_change=float(plateau),
                ray_direction_consistent=bool(half_width+extra<=geometry['upstream_limit'][i]
                    and half_width+extra<=geometry['downstream_limit'][i]),**metrics)
            for tolerance in settings.tolerances:
                label=f'{int(round(100*tolerance)):02d}'
                pv=bool(row['ray_direction_consistent'] and metrics['pressure_mach']>=settings.minimum_mach
                    and metrics['converging_velocity']
                    and metrics['pressure_velocity_relative_error']<=tolerance
                    and plateau<=tolerance)
                # Compatibility of density, entropy and independent fluxes
                # remains distinct from a pressure/velocity-only candidate.
                full=bool(pv and metrics['necessary_shock_conditions']
                    and max(metrics['momentum_relative_error'],metrics['enthalpy_relative_error'],
                            metrics['tangential_jump_over_upstream_sound_speed'],
                            metrics['density_jump_relative_error'])<=tolerance
                    and metrics['entropy_signal_relative_error']<=settings.entropy_signal_tolerance)
                row[f'pressure_velocity_{label}pct']=pv
                row[f'full_jump_{label}pct']=full
            rows.append(row)
    return rows,geometry


def summarize(rows,geometry):
    result=dict(geometric_candidates=len(geometry['indices']),sampled_pairs=len(rows),
        candidates_with_no_valid_window=len(geometry['indices'])-len({r['candidate_index'] for r in rows}))
    for label in ('05','10','20'):
        pv={r['candidate_index'] for r in rows if r[f'pressure_velocity_{label}pct']}
        full={r['candidate_index'] for r in rows if r[f'full_jump_{label}pct']}
        result[f'pressure_velocity_candidates_{label}pct']=len(pv)
        result[f'full_jump_candidates_{label}pct']=len(full)
        result[f'pressure_velocity_only_candidates_{label}pct']=len(pv-full)
    return result
