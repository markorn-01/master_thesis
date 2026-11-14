"""Controlled two-wind collision: global diagnostics and unlabeled surfaces.

No radial shock identities, surface-integrated power, or mesh reconstruction.
The chosen geometry is a first experiment, not a guaranteed collision-time model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from astronomix import time_integration
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping._utils import _unpad
from experiments.wind_bubble.run_single_bubble import build_problem
from experiments.wind_bubble.run_online_bubble import (
    diagnostic_payload, distribution, MACH_EDGES, DENSITY_EDGES,
)
from experiments.wind_bubble.sparse_surfaces import surface_sample_arrays
from experiments.wind_bubble.two_bubble_source_budget import SOURCES, INJECTION_RADIUS, budget_plan
from experiments.wind_bubble.unequal_sources import strength_plan


def source_distances(positions, sources):
    return np.linalg.norm(np.asarray(positions)[:, None, :] - np.asarray(sources)[None, :, :], axis=-1)


def make_catalog(time, fields, centers, dx, sources=SOURCES):
    mask, offsets, normals, mach, flux, rho2, rho1, valid, post, pre = fields
    samples = surface_sample_arrays(mask, offsets, normals, mach, flux, rho2, rho1,
                                    valid, centers, dx, INJECTION_RADIUS,
                                    post_distance=post, pre_distance=pre)
    distances = source_distances(samples['positions'], sources)
    selected = np.isfinite(distances).all(axis=1) & (distances > INJECTION_RADIUS).all(axis=1)
    # Remove single-center fields before exporting a distinct catalog schema.
    for key in ('radii', 'outside_injection', 'upstream_radii', 'downstream_radii',
                'upstream_inside_injection', 'downstream_inside_injection'):
        samples.pop(key)
    catalog = {key: value[selected] for key, value in samples.items()}
    catalog['source_distances'] = distances[selected]  # Geometry, not source ownership.
    known = catalog['sample_positions_valid']
    for side in ('upstream', 'downstream'):
        distances = source_distances(catalog[f'{side}_positions'], sources)
        catalog[f'{side}_inside_sources'] = known[:, None] & (distances <= INJECTION_RADIUS)
        catalog[f'{side}_inside_any_source'] = catalog[f'{side}_inside_sources'].any(axis=1)
    catalog['mach_valid'] = np.isfinite(catalog['mach']) & (catalog['mach'] > 0)
    catalog['thermalization_flux_valid'] = np.isfinite(catalog['thermalization_flux']) & (catalog['thermalization_flux'] > 0)
    catalog.update(schema_version=np.asarray(3, dtype=np.int32),
                   catalog_kind=np.asarray('multi_source_unlabeled_surface_samples'),
                   time=np.asarray(time), grid_shape=np.asarray(np.shape(mask), dtype=np.int32),
                   grid_spacing=np.asarray(dx), source_positions=np.asarray(sources),
                   injection_radius=np.asarray(INJECTION_RADIUS),
                   selection=np.asarray('detected refined surface points outside every injection sphere'),
                   units=np.asarray('dimensionless baseline code units'), mach_estimator=np.asarray('profile-aware'))
    return catalog, int(np.asarray(mask).sum())


class CollisionWriter:
    def __init__(self, directory, centers, dx, sources=SOURCES):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.centers, self.dx, self.sources = np.asarray(centers), dx, sources
        self.rows = []
        self.stream = (self.directory/'history.csv').open('x', newline='')
        self.writer = None

    def receive(self, time, *fields):
        if self.rows and float(time) <= self.rows[-1]['time']:
            raise ValueError('Analysis times must increase strictly.')
        if len(fields) != 12:
            raise ValueError('Expected ten primary grids and two background slices.')
        catalog, total = make_catalog(time, fields[:10], self.centers, self.dx, self.sources)
        count = len(catalog['positions'])
        mh, mi = distribution(catalog['mach'], MACH_EDGES)
        rh, ri = distribution(catalog['density_jump'], DENSITY_EDGES)
        row = dict(time=float(time), surface_cells_all=total, surface_cells_outside_injection=count)
        row.update({f'mach_{key}': value for key, value in mi.items()})
        row.update({f'density_jump_{key}': value for key, value in ri.items()})
        row.update(sample_positions_unknown=int((~catalog['sample_positions_valid']).sum()),
                   upstream_inside_any_source=int(catalog['upstream_inside_any_source'].sum()),
                   downstream_inside_any_source=int(catalog['downstream_inside_any_source'].sum()),
                   minimum_surface_boundary_clearance=float(np.minimum(catalog['positions'], 1-catalog['positions']).min()) if count else np.nan)
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        index = len(self.rows)
        self.writer.writerow(row)
        self.stream.flush()
        np.savez_compressed(self.directory/f'surfaces_{index:04d}.npz', **catalog)
        np.savez_compressed(self.directory/f'histograms_{index:04d}.npz', time=float(time),
                            mach_edges=MACH_EDGES, mach_counts=mh,
                            density_jump_edges=DENSITY_EDGES, density_jump_counts=rh)
        # Small physical background slices help inspect contact and interaction.
        np.savez_compressed(self.directory/f'slice_{index:04d}.npz', time=float(time),
                            density=np.asarray(fields[10]), pressure=np.asarray(fields[11]),
                            z_coordinate=(self.centers.shape[2]//2+.5)*self.dx,
                            grid_spacing=self.dx)
        self.rows.append(row)
        print(f"analysis t={float(time):.5f}: {count} surface samples; Mach median={mi['median']:.5g}", flush=True)

    def close(self):
        self.stream.close()


def compare_outputs(online, offline):
    if len(online.rows) != len(offline.rows) or not online.rows:
        raise AssertionError('Mismatching or empty event sequences.')
    maximum = 0.
    for i, (a, b) in enumerate(zip(online.rows, offline.rows)):
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-8, equal_nan=True, err_msg=key)
            if np.isfinite([a[key], b[key]]).all():
                maximum = max(maximum, abs(a[key]-b[key]))
        for prefix in ('surfaces', 'histograms', 'slice'):
            name = f'{prefix}_{i:04d}.npz'
            with np.load(online.directory/name) as x, np.load(offline.directory/name) as y:
                assert set(x.files) == set(y.files)
                for key in x.files:
                    if x[key].dtype.kind == 'f' and prefix != 'histograms':
                        np.testing.assert_allclose(x[key], y[key], rtol=1e-6, atol=1e-8, equal_nan=True, err_msg=f'{name}: {key}')
                    else:
                        np.testing.assert_array_equal(x[key], y[key], err_msg=f'{name}: {key}')
    return dict(passed=True, compared_events=len(online.rows), fields_per_event=len(online.rows[0]),
                max_absolute_scalar_difference=maximum, histogram_counts_exact=True,
                sparse_surfaces_match=True, background_slices_match=True,
                scalar_rtol=1e-6, scalar_atol=1e-8)


def run(args, *, state_observer=None):
    if args.num_cells < 16 or args.num_cells % 16:
        raise ValueError('num-cells must be a multiple of 16 and at least 16.')
    if args.num_analysis < 3 or not np.isfinite(args.t_end) or args.t_end <= 0:
        raise ValueError('Require at least three events and finite positive t-end.')
    if args.verify_offline and args.num_cells > 64:
        raise ValueError('Stored-state verification is restricted to at most 64³.')
    if args.require_gpu and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU requested but unavailable.')
    setup = argparse.Namespace(num_cells=args.num_cells, num_snapshots=args.num_analysis,
                               t_end=args.t_end, num_injection_cells=args.num_cells//16, plasma_beta=None)
    state, base, params, variables, helper = build_problem(setup)
    strengths=getattr(args,'source_strengths',None)
    base = base._replace(wind_config=base.wind_config._replace(source_positions=SOURCES,
                         source_strengths=tuple(strengths) if strengths is not None else ()),
                         use_specific_snapshot_timepoints=True, print_elapsed_time=False, progress_bar=False)
    base = finalize_config(base, state.shape)
    times = jnp.linspace(0., args.t_end, args.num_analysis)
    params = params._replace(snapshot_timepoints=times, dt_max=args.t_end/args.num_analysis)
    analysis = base._replace(return_snapshots=False, activate_snapshot_callback=False)
    config = analysis._replace(activate_snapshot_callback=True)
    centers, dx = np.asarray(helper.geometric_centers), float(base.grid_spacing)
    injection_counts = [int((np.linalg.norm(centers-np.asarray(s), axis=-1) <= INJECTION_RADIUS-dx/2).sum()) for s in SOURCES]
    if not all(injection_counts):
        raise ValueError('Resolution leaves an empty injection mask.')
    reference_path=getattr(args,'match_source_budget',None)
    reference=json.loads(Path(reference_path).read_text()) if reference_path is not None else None
    budget=budget_plan(args.num_cells,reference,float(params.wind_params.wind_mass_loss_rate),
                       float(params.wind_params.wind_final_velocity),actual_counts=injection_counts)
    weighted=strength_plan(budget,strengths)
    if reference is not None:
        params=params._replace(wind_params=params.wind_params._replace(
            wind_mass_loss_rate=budget['wind_mass_loss_rate_per_source']))
        print(f"Matching reference discrete source budget: mass-parameter scale={budget['parameter_scale']:.9g}; "
              f"prescribed energy rates={budget['discrete_energy_rates']}",flush=True)
    if strengths is not None:
        print(f"Applying source strengths {weighted['source_strengths']}; weighted prescribed energy rates="
              f"{weighted['discrete_energy_rates']}",flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    source_files = [Path(__file__), Path('experiments/shock/analyze_state.py'),
                    Path('experiments/wind_bubble/sparse_surfaces.py'),
                    Path('experiments/wind_bubble/run_online_bubble.py'),
                    Path('astronomix/_modules/_stellar_wind/stellar_wind.py'),
                    Path('astronomix/_modules/_stellar_wind/stellar_wind_options.py'),
                    Path('experiments/wind_bubble/two_bubble_source_budget.py'),
                    Path('experiments/wind_bubble/unequal_sources.py'),
                    Path('astronomix/option_classes/simulation_config.py')]
    metadata = dict(status='running', commit=revision, backend=jax.default_backend(),
                    python=platform.python_version(), jax=jax.__version__, num_cells=args.num_cells,
                    t_end=args.t_end, requested_times=np.asarray(times).tolist(),
                    source_positions=SOURCES, injection_radius=INJECTION_RADIUS,
                    wind_mass_loss_rate_per_source=budget['wind_mass_loss_rate_per_source'], wind_terminal_velocity=budget['wind_terminal_velocity'],
                    nominal_wind_luminosity_per_source=budget['nominal_wind_luminosity_per_source'],
                    nominal_total_wind_luminosity=sum(weighted['nominal_wind_luminosities']),
                    wind_parameter_convention='Shared base mass parameter times source_strengths; velocity is shared.',
                    injection_cell_counts=injection_counts,
                    **weighted,
                    source_budget_plan=budget,
                    source_budget_plan_scope='Equal-source base coefficient before source-strength weighting.',
                    source_budget_reference=dict(path=str(reference_path),sha256=hashlib.sha256(Path(reference_path).read_bytes()).hexdigest()) if reference is not None else None,
                    injection_prescription='existing EI analytic-volume normalization and radius-minus-half-cell mask',
                    ambient_density=1., ambient_pressure=.01, gamma=5/3, domain=[0., 1.],
                    units='dimensionless baseline code units', surface_catalog_schema_version=3,
                    surface_selection='outside both injection spheres using refined positions',
                    source_endpoint_flags='per-source and union, advisory only',
                    source_distances='geometric descriptors, not shock ownership or physical labels',
                    histogram_weighting='one count per selected surface cell',
                    radial_classification=False, temporal_tracking=False, surface_integrated_power=False,
                    online_retains_gas_snapshots=False, verify_offline=args.verify_offline,
                    transfer='ten fixed-shape diagnostic grids plus two 2D slices',
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    meta_path = args.output_dir/'metadata.json'
    meta_path.write_text(json.dumps(metadata, indent=2)+'\n')
    writer = CollisionWriter(args.output_dir/'online', centers, dx)

    def payload(physical):
        return (*diagnostic_payload(physical, analysis, variables, helper, sampling_quality=True),
                physical[variables.density_index, :, :, args.num_cells//2],
                physical[variables.pressure_index, :, :, args.num_cells//2])

    def callback(time, padded, registry):
        physical = _unpad(padded, config)
        jax.debug.callback(writer.receive, time, *payload(physical), ordered=True)
        if state_observer is not None:
            state_observer(time, physical)

    try:
        final = time_integration(state, config, params, variables, snapshot_callable=callback)
        jax.block_until_ready(final)
        jax.effects_barrier()
        writer.close()
        np.testing.assert_allclose([r['time'] for r in writer.rows], times, rtol=0, atol=1e-6)
        if not np.isfinite(np.asarray(final)).all():
            raise AssertionError('Non-finite final gas state.')
        if args.verify_offline:
            stored = time_integration(state, base._replace(activate_snapshot_callback=False), params, variables)
            jax.block_until_ready(stored)
            reference = CollisionWriter(args.output_dir/'offline_reference', centers, dx)
            compiled_payload = jax.jit(payload)
            try:
                for time, physical in zip(np.asarray(stored.time_points), stored.states):
                    reference.receive(time, *compiled_payload(physical))
            finally:
                reference.close()
            result = compare_outputs(writer, reference)
            np.testing.assert_allclose(final, stored.final_state, rtol=1e-6, atol=1e-8)
            result['final_gas_state_matches'] = True
            (args.output_dir/'equivalence.json').write_text(json.dumps(result, indent=2)+'\n')
        metadata.update(status='complete', actual_times=[r['time'] for r in writer.rows])
    except Exception as exc:
        metadata.update(status='failed', error=str(exc))
        raise
    finally:
        writer.close()
        meta_path.write_text(json.dumps(metadata, indent=2)+'\n')
    return writer.rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-cells', type=int, default=64)
    parser.add_argument('--num-analysis', type=int, default=17)
    parser.add_argument('--t-end', type=float, default=.16)
    parser.add_argument('--verify-offline', action='store_true')
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--source-strengths',type=float,nargs=2,default=None,
                        help='Positive per-source mass/luminosity multipliers, e.g. 1 0.5. Default: identical winds.')
    parser.add_argument('--match-source-budget',type=Path,default=None,
                        help='Explicitly adjust the shared mass-rate parameter to match this completed run metadata\'s discrete mass and energy rates.')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/two_bubbles_n064'))
    run(parser.parse_args())


if __name__ == '__main__':
    main()
