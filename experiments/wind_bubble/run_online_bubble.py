"""Small online hydrodynamic shock analysis, with optional offline verification.

Run from the repository root:
python -m experiments.wind_bubble.run_online_bubble --verify-offline

No gas snapshots are retained by the online run. The optional verification run
intentionally retains a small snapshot series and evaluates the host analyzer.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from astronomix import time_integration
from astronomix.time_stepping._utils import _unpad
from experiments.shock.analyze_state import analyze_shock_state, analyze_shock_state_device
from experiments.wind_bubble.run_single_bubble import (
    build_problem, split_radial_shock_candidates, _surface_dissipation_statistics,
    _cumulative_trapezoid_over_detections,
)
from experiments.wind_bubble.sparse_surfaces import (
    SCHEMA_VERSION, surface_sample_arrays, sparse_surface_catalog,
)
from experiments.wind_bubble.online_temporal import (
    TemporalDiagnostics, compare_records, plot_temporal_histories,
)

MACH_EDGES = np.linspace(0., 30., 61)
DENSITY_EDGES = np.linspace(0., 8., 65)


def distribution(values, edges):
    """Histogram with explicit accounting for every input sample."""
    values = np.asarray(values)
    valid = np.isfinite(values) & (values > 0)
    finite = values[valid]
    counts, _ = np.histogram(finite, bins=edges)
    info = dict(valid_count=int(valid.sum()), invalid_count=int((~valid).sum()),
                underflow=int(np.sum(finite < edges[0])), overflow=int(np.sum(finite > edges[-1])))
    quantiles = np.percentile(finite, [16, 50, 84]) if finite.size else [np.nan]*3
    info.update(zip(('p16', 'median', 'p84'), map(float, quantiles)))
    return counts, info


def summarize(time, mask, offsets, direction, mach, flux, rho2, rho1, sampling_valid,
              centers, grid_spacing, injection_radius, *, return_samples=False,
              post_distance=None, pre_distance=None):
    """CPU-only reduction; safe inside a host callback (no JAX dispatch)."""
    mask = np.asarray(mask, dtype=bool)
    samples = surface_sample_arrays(mask, offsets, direction, mach, flux, rho2, rho1,
                                    sampling_valid, centers, grid_spacing, injection_radius,
                                    post_distance=post_distance, pre_distance=pre_distance)
    indices, normals = samples['grid_indices'], samples['normals']
    radii, selection = samples['radii'], samples['outside_injection']
    mach_values, flux_values = samples['mach'], samples['thermalization_flux']
    ratio = samples['density_jump']
    mach_hist, mach_info = distribution(mach_values[selection], MACH_EDGES)
    rho_hist, rho_info = distribution(ratio[selection], DENSITY_EDGES)
    row = dict(time=float(time), surface_cells_all=int(mask.sum()),
               surface_cells_outside_injection=int(selection.sum()))
    row.update({f'mach_{key}': value for key, value in mach_info.items()})
    row.update({f'density_jump_{key}': value for key, value in rho_info.items()})
    _, _, separation = split_radial_shock_candidates(radii, injection_radius, grid_spacing)
    if np.isfinite(separation):
        bands = dict(reverse=selection & (radii <= separation), forward=selection & (radii > separation))
    else:
        bands = dict(reverse=np.zeros(radii.shape, bool), forward=selection)
    for name, band in bands.items():
        radius = float(np.median(radii[band])) if band.any() else np.nan
        stats = _surface_dissipation_statistics(flux_values, normals, grid_spacing,
                                                band, radius, surface_indices=indices)
        row[f'{name}_radius_median'] = radius
        _, band_mach = distribution(mach_values[band], MACH_EDGES)
        row[f'{name}_mach_median'] = band_mach['median']
        row.update({f'{name}_{key}': value for key, value in stats.items()})
    for key in ('surface_area', 'dissipation_rate'):
        values = np.asarray([row[f'{name}_{key}'] for name in bands])
        row[f'total_tracked_{key}'] = float(np.nansum(values)) if np.isfinite(values).any() else np.nan
    if return_samples:
        return row, mach_hist, rho_hist, samples
    return row, mach_hist, rho_hist


class DiagnosticWriter:
    """Stream CSV and per-event histograms; retain only scalar rows in memory."""
    def __init__(self, directory, centers, grid_spacing, injection_radius, *, save_surfaces=False,
                 track_shocks=False, integrate_energy=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.centers = np.asarray(centers)
        self.grid_spacing = grid_spacing
        self.injection_radius = injection_radius
        self.save_surfaces = save_surfaces
        self.temporal = TemporalDiagnostics(self.directory, grid_spacing, injection_radius,
                                            track_shocks=track_shocks, integrate_energy=integrate_energy)
        self.rows = []
        self.quality_rows = []
        self.quality_stream = None
        self.stream = (self.directory/'history.csv').open('x', newline='')
        self.writer = None

    def receive(self, time, *fields):
        if len(fields) not in (8, 10):
            raise ValueError('Expected eight primary grids, optionally followed by two sampling distances.')
        payload = (time, *fields[:8])
        geometry = (dict(post_distance=fields[8], pre_distance=fields[9])
                    if len(fields) == 10 else {})
        row, mach, rho, samples = summarize(*payload, self.centers, self.grid_spacing,
                                            self.injection_radius, return_samples=True, **geometry)
        if self.rows and row['time'] <= self.rows[-1]['time']:
            raise ValueError('Analysis times must advance strictly; duplicate callback detected.')
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.stream.flush()
        index = len(self.rows)
        np.savez_compressed(self.directory/f'histograms_{index:04d}.npz',
                            time=row['time'], mach_edges=MACH_EDGES, mach_counts=mach,
                            density_jump_edges=DENSITY_EDGES, density_jump_counts=rho)
        if self.save_surfaces:
            catalog = sparse_surface_catalog(*payload, self.centers, self.grid_spacing,
                                             self.injection_radius, **geometry)
            np.savez_compressed(self.directory/f'surfaces_{index:04d}.npz', **catalog)
        if geometry:
            quality = sampling_quality_row(row['time'], index, samples,
                                           self.grid_spacing, self.injection_radius)
            if self.quality_stream is None:
                self.quality_stream = (self.directory/'sampling_quality.csv').open('x', newline='')
                self.quality_writer = csv.DictWriter(self.quality_stream, fieldnames=list(quality))
                self.quality_writer.writeheader()
            self.quality_writer.writerow(quality)
            self.quality_stream.flush()
            self.quality_rows.append(quality)
        self.temporal.receive(row['time'], samples, row)
        self.rows.append(row)
        print(f"analysis t={row['time']:.5f}: {row['surface_cells_outside_injection']} surface cells; "
              f"Mach median={row['mach_median']:.5g}", flush=True)

    def close(self):
        self.stream.close()
        if self.quality_stream is not None:
            self.quality_stream.close()
        self.temporal.close()


def sampling_quality_row(time, event_index, samples, grid_spacing, injection_radius):
    """Counts over the unchanged outside-source surface population, per band."""
    selected = samples['outside_injection']
    radii = samples['radii']
    _, _, separation = split_radial_shock_candidates(radii, injection_radius, grid_spacing)
    reverse = selected & (radii <= separation) if np.isfinite(separation) else np.zeros_like(selected)
    forward = selected & ~reverse
    row = dict(time=float(time), event_index=event_index)
    for name, band in [('all', selected), ('reverse', reverse), ('forward', forward)]:
        known = band & samples['sample_positions_valid']
        row.update({f'{name}_surface_samples': int(band.sum()),
                    f'{name}_sample_positions_known': int(known.sum()),
                    f'{name}_sample_positions_unknown': int((band & ~known).sum()),
                    f'{name}_sampling_valid': int((band & samples['sampling_valid']).sum()),
                    f'{name}_upstream_inside_injection': int((known & samples['upstream_inside_injection']).sum()),
                    f'{name}_downstream_inside_injection': int((known & samples['downstream_inside_injection']).sum())})
    return row


def plot_histories(rows, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    times = [r['time'] for r in rows]
    for ax, field, label in zip(axes.flat,
            ('mach_median', 'surface_cells_outside_injection', 'total_tracked_surface_area', 'total_tracked_dissipation_rate'),
            ('Profile-aware Mach (cell median)', 'Surface cells outside injection', 'Tracked shock area', 'Tracked thermalization power')):
        ax.plot(times, [r[field] for r in rows], 'o-')
        ax.set(xlabel='Simulation time', ylabel=label)
        ax.grid(alpha=.25)
    fig.suptitle('Online single-bubble diagnostics')
    fig.savefig(path, dpi=170)
    plt.close(fig)


def verify_products(online, offline):
    """Compare scalar products and histogram counts; fail on discrepancies."""
    if len(online.rows) != len(offline.rows):
        raise AssertionError('Online/offline analysis counts differ.')
    if online.save_surfaces != offline.save_surfaces:
        raise AssertionError('Online/offline sparse-output settings differ.')
    for setting in ('track_shocks', 'integrate_energy'):
        if getattr(online.temporal, setting) != getattr(offline.temporal, setting):
            raise AssertionError(f'Online/offline {setting} settings differ.')
    max_abs = 0.
    for index, (actual, expected) in enumerate(zip(online.rows, offline.rows)):
        if actual.keys() != expected.keys():
            raise AssertionError('Online/offline schemas differ.')
        for key in actual:
            if isinstance(expected[key], int):
                if actual[key] != expected[key]:
                    raise AssertionError(f'Event {index}, {key}: count mismatch')
            else:
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-8,
                                           equal_nan=True, err_msg=f'Event {index}, {key}')
                if np.isfinite([actual[key], expected[key]]).all():
                    max_abs = max(max_abs, abs(actual[key]-expected[key]))
        with np.load(online.directory/f'histograms_{index:04d}.npz') as a, np.load(offline.directory/f'histograms_{index:04d}.npz') as b:
            for key in ('mach_counts', 'density_jump_counts', 'mach_edges', 'density_jump_edges'):
                np.testing.assert_array_equal(a[key], b[key], err_msg=f'Event {index}, {key}')
        if online.save_surfaces:
            with np.load(online.directory/f'surfaces_{index:04d}.npz') as a, np.load(offline.directory/f'surfaces_{index:04d}.npz') as b:
                if set(a.files) != set(b.files):
                    raise AssertionError(f'Event {index}: sparse schemas differ.')
                for key in a.files:
                    if a[key].dtype.kind == 'f':
                        np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-8,
                                                   equal_nan=True, err_msg=f'Event {index}, sparse {key}')
                    else:
                        np.testing.assert_array_equal(a[key], b[key], err_msg=f'Event {index}, sparse {key}')
    verification = dict(passed=True, compared_events=len(online.rows), fields_per_event=len(online.rows[0]),
                max_absolute_scalar_difference=max_abs, histogram_counts_exact=True,
                scalar_rtol=1e-6, scalar_atol=1e-8)
    if online.save_surfaces:
        verification.update(sparse_surfaces_match=True, sparse_surface_events=len(online.rows))
    compare_records(online.quality_rows, offline.quality_rows, 'sampling quality')
    if online.quality_rows:
        verification.update(sampling_quality_matches=True, compared_quality_events=len(online.quality_rows))
    if online.temporal.track_shocks:
        compare_records(online.temporal.track_rows, offline.temporal.track_rows, 'tracking')
        verification.update(tracking_matches=True, compared_track_rows=len(online.temporal.track_rows))
    if online.temporal.integrate_energy:
        compare_records(online.temporal.energy_rows, offline.temporal.energy_rows, 'energy')
        for kind in ('reverse', 'forward'):
            expected_energy, expected_intervals = _cumulative_trapezoid_over_detections(
                np.asarray([r['time'] for r in offline.rows]),
                np.asarray([r[f'{kind}_dissipation_rate'] for r in offline.rows]))
            np.testing.assert_allclose(
                [r[f'{kind}_cumulative_dissipated_energy'] for r in online.temporal.energy_rows],
                expected_energy, rtol=1e-6, atol=1e-8, equal_nan=True)
            np.testing.assert_array_equal(
                [r[f'{kind}_integration_interval_valid'] for r in online.temporal.energy_rows],
                expected_intervals)
        verification.update(energy_matches=True, compared_energy_events=len(online.temporal.energy_rows),
                            offline_batch_energy_matches=True)
    return verification


def diagnostic_payload(physical, config, registry, helper, *, sampling_quality=False):
    """Shared fixed-shape primary fields for online and benchmark workflows."""
    result, mach, flux, samples = analyze_shock_state_device(physical, config, registry, helper)
    _, _, rho2, rho1, valid, post, pre = samples
    payload = (result.shock_surface_cells, result.shock_surface_offsets,
            result.shock_direction, mach, flux, rho2, rho1, valid)
    return (*payload, post, pre) if sampling_quality else payload


def make_online_callback(receive, online_config, analysis_config, helper, *, sampling_quality=False):
    """Build the callback without binding it to a particular output directory."""
    def callback(time, padded, registry):
        physical = _unpad(padded, online_config)
        payload = diagnostic_payload(physical, analysis_config, registry, helper,
                                     sampling_quality=sampling_quality)
        jax.debug.callback(receive, time, *payload, ordered=True)
    return callback


def run(args):
    if args.num_cells < 16 or args.num_cells % 16:
        raise ValueError('num-cells must be a multiple of 16, preserving injection radius 0.0625.')
    if args.num_analysis < 3 or not np.isfinite(args.t_end) or args.t_end <= 0:
        raise ValueError('Require num-analysis >= 3 and finite t-end > 0.')
    if args.verify_offline and args.num_cells > 64:
        raise ValueError('Offline verification retains snapshots; restrict it to at most 64³.')
    if args.require_gpu and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU required, but JAX did not select a GPU backend.')
    # Fail before simulation rather than overwrite a previous experiment.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    setup = argparse.Namespace(num_cells=args.num_cells, num_snapshots=args.num_analysis,
                               t_end=args.t_end, num_injection_cells=args.num_cells//16, plasma_beta=None)
    state, base, params, variables, helper = build_problem(setup)
    times = jnp.linspace(0., args.t_end, args.num_analysis)
    base = base._replace(use_specific_snapshot_timepoints=True, print_elapsed_time=False, progress_bar=False)
    params = params._replace(snapshot_timepoints=times, dt_max=args.t_end/args.num_analysis)
    analysis_config = base._replace(return_snapshots=False, activate_snapshot_callback=False)
    online_config = base._replace(return_snapshots=False, activate_snapshot_callback=True)
    centers = np.asarray(helper.geometric_centers)
    dx = float(base.grid_spacing)
    save_surfaces = getattr(args, 'save_surfaces', False)
    track_shocks = getattr(args, 'track_shocks', False)
    integrate_energy = getattr(args, 'integrate_energy', False)
    online = DiagnosticWriter(args.output_dir/'online', centers, dx, .0625,
                              save_surfaces=save_surfaces, track_shocks=track_shocks,
                              integrate_energy=integrate_energy)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    source_files = [Path(__file__), Path('experiments/shock/analyze_state.py'),
                    Path('experiments/wind_bubble/sparse_surfaces.py'),
                    Path('experiments/wind_bubble/online_temporal.py'),
                    Path('experiments/wind_bubble/run_single_bubble.py')]
    source_hashes = {str(p.relative_to(Path.cwd()) if p.is_absolute() else p):
                     hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    metadata = dict(status='running', schema_version=1, commit=revision, source_sha256=source_hashes,
                    python=platform.python_version(), jax=jax.__version__, backend=jax.default_backend(),
                    num_cells=args.num_cells, injection_radius=.0625, t_end=args.t_end,
                    requested_times=np.asarray(times).tolist(), mach_estimator='profile-aware',
                    density_jump='rho2/rho1 from the same pressure-selected profile samples',
                    units='dimensionless baseline code units', gamma=5/3,
                    ambient_density=1., ambient_pressure=.01, wind_mass_loss_rate=.01,
                    wind_terminal_velocity=10., wind_luminosity=.5,
                    num_injection_cells=args.num_cells//16,
                    density_jump_validity='finite positive samples; not a guarantee of physical compression',
                    online_retains_gas_snapshots=False, verify_offline=args.verify_offline,
                    transfer='fixed-shape diagnostic grids; CPU compression, no dense fields saved',
                    histogram_weighting='surface-cell counts, outside injection region',
                    area_method='existing projected-patch deduplication separately per radial band',
                    radial_labels='single-bubble validation layer, not general shock identities')
    metadata.update(sparse_surfaces=save_surfaces,
                    sparse_surface_schema_version=SCHEMA_VERSION if save_surfaces else None,
                    sparse_surface_selection='detected surface cells outside injection; invalid properties retained',
                    sparse_surface_geometry='refined point samples, no mesh or persistent shock IDs')
    metadata.update(temporal_tracking=track_shocks, cumulative_energy=integrate_energy,
                    temporal_schema_version=1 if (track_shocks or integrate_energy) else None,
                    tracking_method='single-bubble radial candidates with offline continuity and quality flags',
                    tracking_state_update='every detected band, including flagged discontinuities; no reassignment',
                    tracking_mach_estimator='profile-aware',
                    energy_integration='trapezoids over consecutive finite per-band powers; gaps not bridged',
                    energy_continuity_policy='advisory flags, no continuity-based energy filtering')
    meta_path = args.output_dir/'metadata.json'
    meta_path.write_text(json.dumps(metadata, indent=2)+'\n')

    metadata.update(sampling_quality=True, sampling_quality_schema_version=1,
                    sampling_quality_policy='advisory source-endpoint flags; no filtering or energy changes',
                    sampling_quality_transfer='two additional fixed-shape distance grids')
    meta_path.write_text(json.dumps(metadata, indent=2)+'\n')
    callback = make_online_callback(online.receive, online_config, analysis_config, helper,
                                    sampling_quality=True)

    try:
        final = time_integration(state, online_config, params, variables, snapshot_callable=callback)
        jax.block_until_ready(final)
        jax.effects_barrier()
        online.close()
        np.testing.assert_allclose([r['time'] for r in online.rows], np.asarray(times), atol=1e-6, rtol=0)
        if not np.isfinite(np.asarray(final)).all():
            raise AssertionError('Non-finite final gas state.')
        plot_histories(online.rows, args.output_dir/'online_histories.png')
        if track_shocks or integrate_energy:
            plot_temporal_histories(online.temporal, args.output_dir/'online_temporal_histories.png')
        if args.verify_offline:
            # Validation-only run: it intentionally retains complete snapshots.
            snapshots = time_integration(state, base._replace(activate_snapshot_callback=False), params, variables)
            jax.block_until_ready(snapshots)
            offline = DiagnosticWriter(args.output_dir/'offline_reference', centers, dx, .0625,
                                       save_surfaces=save_surfaces, track_shocks=track_shocks,
                                       integrate_energy=integrate_energy)
            # Match the online compilation of the primary sampling calculations.
            # Eager and fused float32 interpolation can differ at individual cells.
            reference_payload = jax.jit(
                lambda physical, h: diagnostic_payload(physical, analysis_config, variables, h,
                                                       sampling_quality=True))
            try:
                for time, physical in zip(np.asarray(snapshots.time_points), np.asarray(snapshots.states)):
                    result = analyze_shock_state(physical, analysis_config, variables, helper)
                    _, offsets, _, _, _, rho2, rho1, valid, post, pre = reference_payload(
                        jnp.asarray(physical), helper)
                    offline.receive(time, result['surface_mask'], np.asarray(offsets),
                                    np.moveaxis(result['shock_direction'], -1, 0),
                                    result['profile_aware_mach_numbers'], result['profile_aware_thermal_energy_flux'],
                                    np.asarray(rho2), np.asarray(rho1), np.asarray(valid),
                                    np.asarray(post), np.asarray(pre))
            finally:
                offline.close()
            verification = verify_products(online, offline)
            np.testing.assert_allclose(np.asarray(final), np.asarray(snapshots.final_state), rtol=1e-6, atol=1e-8)
            verification['final_gas_state_matches'] = True
            (args.output_dir/'equivalence.json').write_text(json.dumps(verification, indent=2)+'\n')
        metadata.update(status='complete', actual_times=[r['time'] for r in online.rows])
        meta_path.write_text(json.dumps(metadata, indent=2)+'\n')
    except Exception as exc:
        online.close()
        metadata.update(status='failed', error=str(exc))
        meta_path.write_text(json.dumps(metadata, indent=2)+'\n')
        raise
    return online.rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-cells', type=int, default=32)
    parser.add_argument('--num-analysis', type=int, default=5, help='Number of explicit times, including initial and final.')
    parser.add_argument('--t-end', type=float, default=.20)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/online_single_bubble_n032'))
    parser.add_argument('--verify-offline', action='store_true')
    parser.add_argument('--save-surfaces', action='store_true',
                        help='Write sparse surface point catalogs at every analysis event.')
    parser.add_argument('--track-shocks', action='store_true',
                        help='Write single-bubble candidate continuity and quality diagnostics.')
    parser.add_argument('--integrate-energy', action='store_true',
                        help='Write cumulative energy from consecutive finite per-band power measurements.')
    parser.add_argument('--require-gpu', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
