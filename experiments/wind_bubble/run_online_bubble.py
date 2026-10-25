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
              centers, grid_spacing, injection_radius):
    """CPU-only reduction; safe inside a host callback (no JAX dispatch)."""
    mask = np.asarray(mask, dtype=bool)
    indices = np.argwhere(mask)
    normals = np.moveaxis(np.asarray(direction), 0, -1)[mask]
    positions = centers[mask] + grid_spacing*np.asarray(offsets)[mask, None]*normals
    radii = np.linalg.norm(positions - np.full(3, .5), axis=-1)
    selection = np.isfinite(radii) & (radii > injection_radius)
    mach_values = np.asarray(mach)[mask]
    flux_values = np.asarray(flux)[mask]
    a, b = np.asarray(rho2)[mask], np.asarray(rho1)[mask]
    sample_valid = np.asarray(sampling_valid)[mask] & np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    ratio = np.full(a.shape, np.nan, dtype=float)
    np.divide(a, b, out=ratio, where=sample_valid)
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
    return row, mach_hist, rho_hist


class DiagnosticWriter:
    """Stream CSV and per-event histograms; retain only scalar rows in memory."""
    def __init__(self, directory, centers, grid_spacing, injection_radius):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.centers = np.asarray(centers)
        self.grid_spacing = grid_spacing
        self.injection_radius = injection_radius
        self.rows = []
        self.stream = (self.directory/'history.csv').open('x', newline='')
        self.writer = None

    def receive(self, *payload):
        row, mach, rho = summarize(*payload, self.centers, self.grid_spacing, self.injection_radius)
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
        self.rows.append(row)
        print(f"analysis t={row['time']:.5f}: {row['surface_cells_outside_injection']} surface cells; "
              f"Mach median={row['mach_median']:.5g}", flush=True)

    def close(self):
        self.stream.close()


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
    return dict(passed=True, compared_events=len(online.rows), fields_per_event=len(online.rows[0]),
                max_absolute_scalar_difference=max_abs, histogram_counts_exact=True,
                scalar_rtol=1e-6, scalar_atol=1e-8)


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
    online = DiagnosticWriter(args.output_dir/'online', centers, dx, .0625)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    source_files = [Path(__file__), Path('experiments/shock/analyze_state.py')]
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
    meta_path = args.output_dir/'metadata.json'
    meta_path.write_text(json.dumps(metadata, indent=2)+'\n')

    def callback(time, padded, registry):
        physical = _unpad(padded, online_config)
        result, mach, flux, samples = analyze_shock_state_device(physical, analysis_config, registry, helper)
        _, _, rho2, rho1, valid, _, _ = samples
        jax.debug.callback(online.receive, time, result.shock_surface_cells,
                           result.shock_surface_offsets, result.shock_direction, mach, flux,
                           rho2, rho1, valid, ordered=True)

    try:
        final = time_integration(state, online_config, params, variables, snapshot_callable=callback)
        jax.block_until_ready(final)
        jax.effects_barrier()
        online.close()
        np.testing.assert_allclose([r['time'] for r in online.rows], np.asarray(times), atol=1e-6, rtol=0)
        if not np.isfinite(np.asarray(final)).all():
            raise AssertionError('Non-finite final gas state.')
        plot_histories(online.rows, args.output_dir/'online_histories.png')
        if args.verify_offline:
            # Validation-only run: it intentionally retains complete snapshots.
            snapshots = time_integration(state, base._replace(activate_snapshot_callback=False), params, variables)
            jax.block_until_ready(snapshots)
            offline = DiagnosticWriter(args.output_dir/'offline_reference', centers, dx, .0625)
            try:
                for time, physical in zip(np.asarray(snapshots.time_points), np.asarray(snapshots.states)):
                    result = analyze_shock_state(physical, analysis_config, variables, helper)
                    device_result, _, _, samples = analyze_shock_state_device(
                        jnp.asarray(physical), analysis_config, variables, helper)
                    _, _, rho2, rho1, valid, _, _ = samples
                    offsets = device_result.shock_surface_offsets
                    offline.receive(time, result['surface_mask'], np.asarray(offsets),
                                    np.moveaxis(result['shock_direction'], -1, 0),
                                    result['profile_aware_mach_numbers'], result['profile_aware_thermal_energy_flux'],
                                    np.asarray(rho2), np.asarray(rho1), np.asarray(valid))
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
    parser.add_argument('--require-gpu', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
