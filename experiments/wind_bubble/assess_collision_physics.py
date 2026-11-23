"""Independent, read-only Euler jump checks on captured collision states.

State-pair compatibility is necessary, not sufficient, to identify a shock.
No production normal, shock mask, Mach threshold, grouping or meshing is used.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


ROOT = Path('outputs/three_bubble_collision_detection_audit_iwr_compgpu8_n064')
SOURCES = np.array([[.375,.375,.5],[.625,.375,.5],[.5,.625,.375]])
BOUNDS = np.array([[.45,.55],[.335,.415],[.42,.58]])
HALF_WIDTHS = (1,2,3)
CONTRAST_FLOOR = 1.e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_metrics(up, down, normal, gamma=5/3):
    """Necessary RH/entropy checks for two specified primitive states.

    Normal points from proposed upstream to downstream. Speed is inferred
    from mass conservation, so its mass residual is not independent evidence.
    Momentum, total enthalpy, tangential velocity and entropy remain tests.
    """
    a,b,n = np.asarray(up,float),np.asarray(down,float),np.asarray(normal,float)
    if (a.shape!=(5,) or b.shape!=(5,) or n.shape!=(3,) or not np.isfinite([*a,*b,*n]).all()
            or min(a[0],b[0],a[4],b[4])<=0 or gamma<=1 or not np.isclose(n@n,1)):
        raise ValueError('Require finite positive primitive states and a unit 3D normal.')
    rho1,rho2,p1,p2 = a[0],b[0],a[4],b[4]
    u1,u2 = float(a[1:4]@n),float(b[1:4]@n)
    c1,c2 = np.sqrt(gamma*p1/rho1),np.sqrt(gamma*p2/rho2)
    r,p = rho2/rho1,p2/p1
    k = p/r**gamma
    e1,e2 = p1/((gamma-1)*rho1),p2/((gamma-1)*rho2)
    work = .5*(p1+p2)*(1/rho1-1/rho2)
    thermal_residual = (e2-e1-work)/max(e1,e2,abs(work))
    speed = ((rho2*u2-rho1*u1)/(rho2-rho1)
             if abs(rho2-rho1)>1.e-12*max(rho1,rho2) else np.nan)
    w1,w2 = u1-speed,u2-speed
    vt1,vt2 = a[1:4]-u1*n,b[1:4]-u2*n
    mom1,mom2 = p1+rho1*w1*w1,p2+rho2*w2*w2
    h1 = gamma/(gamma-1)*p1/rho1+.5*(w1*w1+vt1@vt1)
    h2 = gamma/(gamma-1)*p2/rho2+.5*(w2*w2+vt2@vt2)
    momentum_error = abs(mom2-mom1)/max(mom1,mom2)
    enthalpy_error = abs(h2-h1)/max(h1,h2)
    tangent_error = np.linalg.norm(vt2-vt1)/c1
    contrast = bool(r>1+CONTRAST_FLOOR and p>1+CONTRAST_FLOOR)
    converging = bool(u1-u2>1.e-10*c1)
    entropy_ok = bool(np.log(k)>1.e-10)
    # Necessary gas-dynamic shock-frame direction and sonic ordering.
    sonic_ok = bool(np.isfinite(speed) and w1>c1 and 0<w2<c2)
    admissible = contrast and converging and entropy_ok and sonic_ok
    result = dict(density_ratio=r,pressure_ratio=p,temperature_ratio=p/r,entropy_ratio=k,
        entropy_change_over_cv=np.log(k),normal_velocity_drop=u1-u2,
        internal_energy_change=e2-e1,hugoniot_work=work,
        thermal_hugoniot_relative_residual=thermal_residual,
        mass_inferred_speed=speed,upstream_shock_frame_mach=w1/c1,
        downstream_shock_frame_mach=w2/c2,
        momentum_relative_error=momentum_error,enthalpy_relative_error=enthalpy_error,
        tangential_jump_over_upstream_sound_speed=tangent_error,
        compressive_contrast=contrast,converging_velocity=converging,
        entropy_increase=entropy_ok,sonic_ordering=sonic_ok,necessary_shock_conditions=admissible)
    for tolerance in (.05,.10,.20):
        result[f'compatible_{int(tolerance*100):02d}pct'] = bool(admissible and
            max(momentum_error,enthalpy_error,tangent_error)<tolerance)
    return result


def source_free_segments(a,b,sources=SOURCES,radius=.0625):
    """Reject an entire segment intersecting a source sphere, not only endpoints."""
    a,b = np.asarray(a,float),np.asarray(b,float)
    v = b-a
    t = np.clip(((sources[None]-a[:,None])*v[:,None]).sum(-1)/
                np.maximum((v*v).sum(-1)[:,None],1.e-30),0,1)
    closest = a[:,None]+t[...,None]*v[:,None]
    return (np.linalg.norm(closest-sources[None],axis=-1)>radius).all(axis=1)


def axis_pairs(state,x,time,event,ys,zs,region,outer_control=False):
    rows=[]
    for j,k in itertools.product(ys,zs):
        # One side at a time: never treat both opposing flows as one shock.
        windows = ((.421875,.5),(.5,.578125))
        if outer_control:
            possible=np.flatnonzero((x>=.05)&(x<=.375-.0625))
            peak=possible[np.argmax(state[0,possible,j,k])]
            # Preserve the same physical window when resolution changes.
            windows=((max(.05,x[peak]-6/64),min(.375-.0625,x[peak]+3/64+1.e-9)),)
        for side,(lo,hi) in enumerate(windows):
            ids=np.flatnonzero((x>=lo)&(x<hi))
            for i,l in itertools.combinations(ids,2):
                a=np.array([x[i],x[j],x[k]])
                b=np.array([x[l],x[j],x[k]])
                if not source_free_segments(a[None],b[None])[0]:continue
                q1,q2 = state[:,i,j,k],state[:,l,j,k]
                n=np.array([1.,0.,0.])
                # Higher pressure defines proposed downstream, independently of T.
                if q2[4]<q1[4]:q1,q2,a,b,n=q2,q1,b,a,-n
                row=dict(event=event,time=time,region=region,side=side,
                    upstream_x=a[0],downstream_x=b[0],y=x[j],z=x[k],
                    separation_cells=abs(i-l),**pair_metrics(q1,q2,n))
                rows.append(row)
    return rows


def pressure_rays(state,xyz,time,event,dx,half_widths=HALF_WIDTHS):
    roi=((xyz>=BOUNDS[:,0])&(xyz<=BOUNDS[:,1])).all(-1)
    centers=xyz[roi]
    gradient=np.stack(np.gradient(state[4],dx),-1)[roi]
    magnitude=np.linalg.norm(gradient,axis=1)
    normals=gradient/np.maximum(magnitude[:,None],1.e-30)
    rows=[]
    for half in half_widths:
        a=centers-half*dx*normals
        b=centers+half*dx*normals
        valid=(magnitude>1.e-10)&source_free_segments(a,b)
        valid&=(a>=dx/2).all(-1)&(b>=dx/2).all(-1)&(a<=1-dx/2).all(-1)&(b<=1-dx/2).all(-1)
        def sample(points):
            coords=(points/dx-.5).T
            return np.stack([map_coordinates(field,coords,order=1,mode='nearest') for field in state],axis=1)
        q1,q2=sample(a),sample(b)
        for i in np.flatnonzero(valid):
            row=dict(event=event,time=time,half_width_cells=half,
                center_x=centers[i,0],center_y=centers[i,1],center_z=centers[i,2],
                normal_x=normals[i,0],normal_y=normals[i,1],normal_z=normals[i,2],
                **pair_metrics(q1[i],q2[i],normals[i]))
            rows.append(row)
    return rows


def summarize(rows):
    contrast=[r for r in rows if r['compressive_contrast']]
    flow=[r for r in contrast if r['converging_velocity']]
    thermodynamic=[r for r in flow if r['entropy_increase']]
    admissible=[r for r in rows if r['necessary_shock_conditions']]
    result=dict(sample_pairs=len(rows),compressive_contrast_pairs=len(contrast),
        contrast_and_velocity_pairs=len(flow),contrast_velocity_entropy_pairs=len(thermodynamic),
        necessary_shock_condition_pairs=len(admissible))
    for key in ('compatible_05pct','compatible_10pct','compatible_20pct'):
        result[key]=sum(r[key] for r in rows)
    result['maximum_entropy_ratio_in_compressive_pairs']=max((r['entropy_ratio'] for r in contrast),default=None)
    return result


def write_csv(path,rows):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def make_figures(root,output,meta,events):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    n=meta['num_cells'];x=(np.arange(n)+.5)/n
    iy=int(np.argmin(abs(x-.375)));iz=n//2
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),dpi=150)
    fig.subplots_adjust(left=.07,right=.97,bottom=.20,top=.87,hspace=.42,wspace=.34)
    for event in events:
        state=np.load(root/'audit'/f'state_{event:04d}.npy')
        rho,u,v,w,p=state[:,:,iy,iz]
        fields=(rho,p,p/rho,p/rho**meta['gamma'],u,np.sqrt(meta['gamma']*p/rho))
        for ax,field,title in zip(axes.flat,fields,
                ('Density','Pressure','Temperature proxy P/ρ','Entropy proxy P/ρᵞ','x velocity (lab frame)','Sound speed')):
            ax.plot(x,field,'o-',markersize=2.5,linewidth=1,label=f't={meta["actual_times"][event]:.2f}')
            ax.set(xlim=(.42,.58),xlabel='x',ylabel=title,title=title)
            ax.axvline(.5,color='gray',ls=':',lw=.8);ax.grid(alpha=.2)
            if title=='Entropy proxy P/ρᵞ':ax.set_yscale('log')
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle(f'Independent physical check of the dense collision layer | saved {n}³ states',fontsize=15)
    fig.text(.07,.045,'Entropy proxy decreases into the later dense layer. Connecting its hot exterior and cold interior as one shock\n'
        'would violate the entropy condition. These are spatial profiles, not entropy histories of the same gas parcel.\n'
        'No detector criteria changed. Lab-frame subsonic flow alone is not proof that a moving shock is absent.',fontsize=10)
    fig.savefig(output/'collision_entropy_profiles.png');plt.close(fig)

    # Time-position views show the saved gas structure; no ridge is labeled a shock trajectory.
    times=[];rho=[];pressure=[]
    for index,time in enumerate(meta['actual_times']):
        with np.load(root/'online'/f'slice_{index:04d}.npz') as data:
            times.append(float(data['time']));rho.append(data['density'][:,iy]);pressure.append(data['pressure'][:,iy])
    times=np.asarray(times);rho=np.asarray(rho);pressure=np.asarray(pressure)
    te=np.r_[times[0]-(times[1]-times[0])/2,(times[:-1]+times[1:])/2,times[-1]+(times[-1]-times[-2])/2]
    xe=np.arange(n+1)/n
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),dpi=150)
    fig.subplots_adjust(left=.07,right=.97,top=.80,bottom=.25,wspace=.38)
    selection=(x>=.42)&(x<=.58)
    for ax,values,title in zip(axes,(rho,pressure,pressure/rho),('Density','Pressure','Temperature proxy P/ρ')):
        selected=values[:,selection]
        image=ax.pcolormesh(xe,te,values,cmap='magma',norm=LogNorm(selected.min(),selected.max()),shading='flat')
        ax.set(xlim=(.42,.58),ylim=(0,.16),xlabel='x',ylabel='Simulation time',title=title)
        ax.axvline(.5,color='cyan',ls=':',lw=.8)
        fig.colorbar(image,ax=ax,pad=.02)
    fig.suptitle('Evolution across the lower-source interaction | original saved slices',fontsize=14)
    fig.text(.07,.065,'Incoming structures meet near x=0.5; a central dense layer remains. Color ridges are not independently\n'
        'identified shock fronts. Full 3D states are available only at the five audited times.',fontsize=10)
    fig.savefig(output/'collision_time_position.png');plt.close(fig)


def run(args):
    root=args.root
    meta=json.loads((root/'metadata.json').read_text())
    audit=json.loads((root/'audit/verification.json').read_text())
    if not audit['passed'] or not np.array_equal(meta['source_positions'],SOURCES):
        raise ValueError('Require the verified matching three-bubble audit.')
    if args.output_dir.exists():raise ValueError('Use a fresh output directory.')
    paths=[root/'metadata.json',root/'audit/verification.json']
    events=audit['captured_events']
    for event in events:
        path=root/'audit'/f'state_{event:04d}.npy'
        expected=next(h for p,h in audit['state_sha256'].items() if Path(p).name==path.name)
        if sha(path)!=expected:raise ValueError(f'Captured state changed: {path}')
        paths.append(path)
    paths.extend(root/'online'/f'slice_{i:04d}.npz' for i in range(len(meta['actual_times'])))
    fingerprints={str(p):sha(p) for p in paths}
    n=meta['num_cells'];dx=1/n;x=(np.arange(n)+.5)*dx
    xyz=np.stack(np.meshgrid(x,x,x,indexing='ij'),axis=-1)
    ys=np.flatnonzero((x>=BOUNDS[1,0])&(x<=BOUNDS[1,1]))
    zs=np.flatnonzero((x>=BOUNDS[2,0])&(x<=BOUNDS[2,1]))
    near_y=np.sort(np.argsort(abs(x-.375),kind='stable')[:2])
    near_z=np.sort(np.argsort(abs(x-.5),kind='stable')[:2])
    # Fixed-cell and physically matched widths: N64 {1,2,3}, N128 {1,2,3,4,6}.
    half_widths=tuple(sorted(set(HALF_WIDTHS)|{width*n/64 for width in HALF_WIDTHS}))
    all_axis,all_rays,counts=[],[],[]
    for event in events:
        state=np.load(root/'audit'/f'state_{event:04d}.npy').astype(float)
        if state.shape!=(5,n,n,n) or not np.isfinite(state).all():raise ValueError('Invalid captured primitive state.')
        time=float(meta['actual_times'][event])
        for name,j,k,outer in [('near_axis',near_y,near_z,False),('central_volume',ys,zs,False),
                               ('outer_forward_control',near_y,near_z,True)]:
            rows=axis_pairs(state,x,time,event,j,k,name,outer)
            all_axis.extend(rows);counts.append(dict(event=event,time=time,method=name,**summarize(rows)))
        rows=pressure_rays(state,xyz,time,event,dx,half_widths);all_rays.extend(rows)
        for half in half_widths:
            selected=[r for r in rows if r['half_width_cells']==half]
            counts.append(dict(event=event,time=time,method=f'pressure_normal_half_width_{half:g}',**summarize(selected)))
        print('Physical state-pair checks:',event,flush=True)
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir/'axis_state_pairs.csv',all_axis)
    write_csv(args.output_dir/'pressure_normal_state_pairs.csv',all_rays)
    write_csv(args.output_dir/'summary.csv',counts)
    make_figures(root,args.output_dir,meta,events)
    for path,expected in fingerprints.items():
        if sha(path)!=expected:raise AssertionError('Read-only analysis input changed.')
    report=dict(analysis_completed=True,production_changed=False,new_simulation_run=False,
        input_sha256=fingerprints,source_sha256={str(Path(__file__)):sha(__file__)},
        events=events,summary=counts,normal_sources=['Cartesian x on each side separately','positive pressure gradient'],
        pressure_ray_half_widths_cells=list(half_widths),
        pressure_ray_half_widths_domain_units=[width*dx for width in half_widths],contrast_floor=CONTRAST_FLOOR,
        flux_error_tolerances=[.05,.10,.20],mass_speed_scope='Inferred from each pair, not independently tracked; zero mass residual is tautological.',
        entropy_scope='Spatial entropy contrast across proposed shock states; not Lagrangian entropy production.',
        physical_shock_absence_proven=False,missed_physical_shock_proven=False,
        scope='Necessary single-shock state-pair consistency checks; no unique shock trajectory or guaranteed clean plateau states.')
    (args.output_dir/'assessment.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(analysis_completed=True,production_changed=False,summary=counts),indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args())
