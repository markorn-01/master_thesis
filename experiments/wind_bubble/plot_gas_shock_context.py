"""One saved collision slice with physical gas context and detected shock samples."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def prepare(density,pressure,positions,z,dx):
    density=np.asarray(density,float);pressure=np.asarray(pressure,float);positions=np.asarray(positions,float)
    if density.ndim!=2 or pressure.shape!=density.shape:raise ValueError('Require matching 2D gas slices.')
    if not np.isfinite(density).all() or not np.isfinite(pressure).all() or (density<=0).any() or (pressure<=0).any():
        raise ValueError('Density and pressure must be finite and positive.')
    if positions.ndim!=2 or positions.shape[1]!=3 or not np.isfinite(positions).all():raise ValueError('Require finite 3D positions.')
    if not np.isfinite(z) or not np.isfinite(dx) or dx<=0:raise ValueError('Require a finite plane and positive grid spacing.')
    temperature=pressure/density
    if not np.isfinite(temperature).all():raise ValueError('Nonfinite temperature proxy.')
    rows=np.flatnonzero(np.abs(positions[:,2]-z)<=dx/2)
    return temperature,rows


def run(root,event,output):
    source=root/'online'/f'surfaces_{event:04d}.npz';slice_path=root/'online'/f'slice_{event:04d}.npz';meta_path=root/'metadata.json'
    paths=[source,slice_path,meta_path];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    meta=json.loads(meta_path.read_text())
    with np.load(source) as a:catalog={k:a[k].copy() for k in ['positions','source_positions','injection_radius','grid_spacing','time','grid_shape']}
    with np.load(slice_path) as a:gas={k:a[k].copy() for k in a.files}
    dx=float(gas['grid_spacing']);z=float(gas['z_coordinate']);time=float(gas['time'])
    np.testing.assert_allclose(time,float(catalog['time']),atol=1e-7,rtol=0)
    np.testing.assert_allclose(dx,float(catalog['grid_spacing']),atol=0,rtol=0)
    assert list(gas['density'].shape)==catalog['grid_shape'][:2].tolist()
    domain=meta['domain'];assert domain==[0.,1.]
    assert abs(gas['density'].shape[0]*dx-1)<1e-12 and abs(gas['density'].shape[1]*dx-1)<1e-12
    temperature,rows=prepare(gas['density'],gas['pressure'],catalog['positions'],z,dx)
    output.mkdir(parents=True,exist_ok=False)
    # Display arrays preserve native x/y order; rendering alone transposes for imshow.
    np.savez_compressed(output/'display_data.npz',density=gas['density'],pressure=gas['pressure'],temperature_proxy=temperature,
                        overlay_positions=catalog['positions'][rows],overlay_catalog_rows=rows,
                        time=gas['time'],z_coordinate=gas['z_coordinate'],grid_spacing=gas['grid_spacing'])
    limits=render(gas['density'],temperature,catalog['positions'][rows],catalog['source_positions'],
                  float(catalog['injection_radius']),time,z,dx,output)
    with np.load(output/'display_data.npz') as a:
        np.testing.assert_array_equal(a['density'],gas['density']);np.testing.assert_array_equal(a['pressure'],gas['pressure'])
        np.testing.assert_allclose(a['temperature_proxy'],gas['pressure'].astype(float)/gas['density'].astype(float),rtol=1e-14)
        expected=np.array([i for i,p in enumerate(catalog['positions']) if z-dx/2<=float(p[2])<=z+dx/2],int)
        np.testing.assert_array_equal(rows,expected)
        np.testing.assert_array_equal(a['overlay_positions'],catalog['positions'][expected])
    from PIL import Image
    with Image.open(output/'gas_shock_context.png') as image:size=list(image.size);image.verify()
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    report=dict(event=event,time=time,plane_z=z,grid_spacing=dx,overlay_half_width=dx/2,
        total_catalog_samples=len(catalog['positions']),overlay_samples=len(rows),temperature_definition='P/rho; temperature proxy in code units, not Kelvin',
        slice_array_order='x,y; transposed only for display',image_origin='lower',image_extent=[0,1,0,1],image_interpolation='nearest',
        limits=limits,image_size=size,input_sha256=hashes,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        all_selected_points_displayed=True,positions_moved=False,gas_fields_smoothed=False,three_dimensional_volume_rendered=False,
        shock_detection_changed=False,tracking_changed=False,physical_merger_inferred=False)
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('sha256')},indent=2))


def render(density,temperature,points,sources,injection_radius,time,z,dx,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle
    dlim=[float(density.min()),float(density.max())];tlim=[float(temperature.min()),float(temperature.max())]
    fig,axes=plt.subplots(1,3,figsize=(16,6))
    for index,ax in enumerate(axes):
        field=density if index==0 else temperature;limits=dlim if index==0 else tlim
        image=ax.imshow(field.T,origin='lower',extent=(0,1,0,1),interpolation='nearest',
                        norm=LogNorm(limits[0],max(limits[1],limits[0]*1.01)),cmap='cividis' if index==0 else 'inferno')
        for source in sources:
            ax.scatter(source[0],source[1],marker='*',s=95,c='white',edgecolors='black',linewidths=.7,zorder=5)
            dz=z-source[2]
            if abs(dz)<injection_radius:
                ax.add_patch(Circle(source[:2],np.sqrt(injection_radius**2-dz**2),fill=False,edgecolor='white',linestyle=':',linewidth=1,alpha=.8))
        ax.set(xlim=(.1,.9),ylim=(.1,.9),aspect='equal',xlabel='x',ylabel='y',
               title=['Gas density','Temperature proxy: P/ρ','Same temperature + detected shocks'][index])
        label='Density (code units)' if index==0 else 'P/ρ (code units)'
        cb=fig.colorbar(image,ax=ax,location='bottom',pad=.13,fraction=.05);cb.set_label(label)
    axes[2].scatter(points[:,0],points[:,1],s=9,facecolors='none',edgecolors='#52e7ed',linewidths=.65,
                    label=f'Shock samples near plane ({len(points)})',zorder=4)
    # Keep the overlay legend inside the lower part of the panel to avoid its title.
    axes[2].legend(loc='lower center',fontsize=8,framealpha=.9,facecolor='#202020',labelcolor='white')
    fig.suptitle(f'Two-bubble gas structure and shock detections | t = {time:.2f}',fontsize=17)
    fig.text(.03,.025,f'Saved xy slice: z={z:.6f}. Cyan circles: original shock samples within ±{dx/2:.6f} of that plane (half a grid cell).\n'
             'Stars: projected wind-source centers. Dotted circles: source-region intersections. Logarithmic colors; temperature panels use identical limits.\n'
             'P/ρ is a temperature proxy in code units. This is a 2D slice with a thin-slab overlay, not a 3D volume rendering or exact shock-plane intersection.',fontsize=9)
    fig.subplots_adjust(left=.045,right=.98,top=.85,bottom=.25,wspace=.24)
    fig.savefig(output/'gas_shock_context.png',dpi=140);plt.close(fig)
    return dict(density=dlim,temperature_proxy=tlim,temperature_limits_identical=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/two_bubbles_budget_matched_iwr_compgpu3_n128'))
    parser.add_argument('--event-index',type=int,default=8);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.event_index,args.output_dir)
