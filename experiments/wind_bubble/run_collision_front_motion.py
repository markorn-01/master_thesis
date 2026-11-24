"""Capture frequent small strips for a measured-speed collision-front test.

Designed for one 128³ run on a 12 GiB GPU. No full gas snapshots, production
Mach grids, grouping, or display meshes are retained. Existing production
stage masks are sampled from full-grid calculations for comparison only.
"""
import argparse
from dataclasses import asdict
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
from experiments.wind_bubble.audit_detection_stages import stage_fields
from experiments.wind_bubble.run_single_bubble import build_problem
from experiments.wind_bubble.three_bubble_setup import SOURCES, source_budget
from experiments.wind_bubble.collision_front_motion import MotionSettings


REFERENCE = Path('outputs/three_bubble_collision_resolution_iwr_compgpu8_n128/metadata.json')
STAGES = ('compression', 'gradient_alignment', 'pressure_jump_pass',
          'temperature_jump_pass', 'shock_zone', 'shock_surface')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strip_indices(n, center):
    x = (np.arange(n)+.5)/n
    lower = np.searchsorted(x, center, side='right')-1
    indices = np.arange(lower-1, lower+3)
    if indices[0] < 0 or indices[-1] >= n:
        raise ValueError('Transverse strip is too close to a boundary.')
    return indices


def experiment_plan(reference_path, n, count, t_end):
    if n not in (32, 64, 128):
        raise ValueError('Use 128 cells for the physical check; 32/64 are supported for controls.')
    if count < 9 or not np.isfinite(t_end) or not 0 < t_end <= .16:
        raise ValueError('Require at least nine events and 0 < t-end <= 0.16.')
    reference = json.loads(reference_path.read_text())
    if reference.get('status') != 'complete':
        raise ValueError('Require the completed collision-resolution reference.')
    np.testing.assert_array_equal(reference['source_positions'], SOURCES)
    np.testing.assert_array_equal(reference['source_strengths'], [1., 1., 1.])
    for key, value in (('injection_radius', .0625), ('gamma', 5/3), ('ambient_density', 1.),
                       ('ambient_pressure', .01), ('wind_terminal_velocity', 10.)):
        if reference[key] != value:
            raise ValueError(f'Unexpected reference {key}.')
    budget_path = Path(reference['source_budget_reference']['path'])
    if sha(budget_path) != reference['source_budget_reference']['sha256']:
        raise ValueError('Original source-budget reference changed.')
    budget = source_budget(n, json.loads(budget_path.read_text()))
    for key in ('discrete_mass_rates', 'discrete_energy_rates'):
        np.testing.assert_allclose(budget[key], reference[key], rtol=1.e-12, atol=1.e-12)
    return reference, budget, np.linspace(0, t_end, count, dtype=np.float32), budget_path


class StripWriter:
    def __init__(self, directory, times, n):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.requested = np.asarray(times)
        self.actual = []
        self.n = n
        self.hashes = {}

    def receive(self, time, strip, stages, physical_valid):
        index = len(self.actual)
        if (index >= len(self.requested) or not np.isclose(float(time), self.requested[index], rtol=0, atol=1.e-6)
                or self.actual and float(time) <= self.actual[-1]):
            raise ValueError('Unexpected or repeated strip event.')
        strip, stages = np.asarray(strip), np.asarray(stages)
        if (strip.shape != (5, self.n, 4, 4) or stages.shape != (len(STAGES), self.n, 4, 4)
                or not bool(physical_valid) or not np.isfinite(strip).all()
                or (strip[[0, 4]] <= 0).any()):
            raise ValueError('Invalid physical strip or full-domain state.')
        path = self.directory/f'strip_{index:04d}.npz'
        np.savez_compressed(path, time=float(time), primitive=strip, production_stages=stages)
        self.hashes[path.name] = sha(path)
        self.actual.append(float(time))
        if index % 20 == 0 or index == len(self.requested)-1:
            print(f'flow strip {index+1}/{len(self.requested)}: t={float(time):.5f}', flush=True)


def run(args):
    if args.analyze_only:
        from experiments.wind_bubble.review_collision_front_motion import run as review
        return review(args.output_dir, args.analysis_dir)
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory. --analyze-only reuses a completed integration.')
    if (args.require_gpu or args.num_cells >= 64) and jax.default_backend() != 'gpu':
        raise RuntimeError('This resolution requires a working GPU. Use 32 only for CPU smoke checks.')
    reference, budget, times, budget_path = experiment_plan(
        args.reference_metadata, args.num_cells, args.num_events, args.t_end)
    n = args.num_cells
    state, config, params, registry, helper = build_problem(argparse.Namespace(
        num_cells=n, num_snapshots=len(times), t_end=args.t_end,
        num_injection_cells=n//16, plasma_beta=None))
    if (registry.density_index != 0 or tuple(registry.velocity_index) != (1, 2, 3)
            or registry.pressure_index != 4):
        raise ValueError('Unexpected primitive-state ordering.')
    config = config._replace(wind_config=config.wind_config._replace(source_positions=SOURCES,
        source_strengths=(1., 1., 1.)), return_snapshots=False, activate_snapshot_callback=True,
        use_specific_snapshot_timepoints=True, print_elapsed_time=False, progress_bar=False)
    config = finalize_config(config, state.shape)
    params = params._replace(snapshot_timepoints=jnp.asarray(times), dt_max=reference['solver_dt_max'],
        wind_params=params.wind_params._replace(wind_mass_loss_rate=budget['wind_mass_loss_rate_per_source']))
    iy, iz = strip_indices(n, .375), strip_indices(n, .5)
    source_files = sorted(Path('astronomix').rglob('*.py')) + [
        Path(__file__), Path('experiments/wind_bubble/run_single_bubble.py'),
        Path('experiments/wind_bubble/audit_detection_stages.py'),
        Path('experiments/wind_bubble/three_bubble_setup.py'),
        Path('experiments/wind_bubble/two_bubble_source_budget.py'),
        Path('experiments/wind_bubble/collision_front_motion.py'),
        Path('experiments/wind_bubble/review_collision_front_motion.py')]
    source_hashes = {str(p): sha(p) for p in source_files}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    args.output_dir.mkdir(parents=True)
    meta = dict(status='running', commit=commit, num_cells=n, t_end=args.t_end,
        backend=jax.default_backend(), devices=[str(device) for device in jax.devices()],
        python=platform.python_version(), jax=jax.__version__, gamma=5/3,
        source_positions=SOURCES, source_strengths=[1., 1., 1.], injection_radius=.0625,
        discrete_mass_rates=budget['discrete_mass_rates'], discrete_energy_rates=budget['discrete_energy_rates'],
        source_budget_plan=budget, reference_metadata=str(args.reference_metadata),
        reference_sha256=sha(args.reference_metadata), budget_reference=str(budget_path),
        budget_reference_sha256=sha(budget_path), requested_times=times.tolist(),
        solver_dt_max=reference['solver_dt_max'], schedule_changed=True,
        schedule_scope='Dense strip output adds solver event boundaries; not same-state replay of the old sparse schedule.',
        transverse_y=((iy+.5)/n).tolist(), transverse_z=((iz+.5)/n).tolist(),
        target_y=.375, target_z=.5, stage_names=STAGES,
        strip_shape=[5, n, 4, 4], uncompressed_primitive_payload_bytes=len(times)*5*n*16*4,
        source_sha256=source_hashes, production_changed=False,
        full_snapshots_retained=False, same_state_online_offline_comparison=False,
        purpose='Measure pressure-front motion independently, test shock-frame Euler jumps, compare production masks.',
        motion_settings=asdict(MotionSettings()))
    meta_path = args.output_dir/'metadata.json'
    meta_path.write_text(json.dumps(meta, indent=2)+'\n')
    writer = StripWriter(args.output_dir/'strips', times, n)
    def take_strip(array):
        return array[..., :, iy[0]:iy[-1]+1, iz[0]:iz[-1]+1]
    def callback(time, padded, _registry):
        physical = _unpad(padded, config)
        fields = stage_fields(physical, config, registry, helper)
        stages = jnp.stack([take_strip(fields[key]) for key in STAGES]).astype(jnp.uint8)
        valid = jnp.all(jnp.isfinite(physical)) & jnp.all(physical[0] > 0) & jnp.all(physical[4] > 0)
        jax.debug.callback(writer.receive, time, take_strip(physical), stages, valid, ordered=True)
    try:
        final = time_integration(state, config, params, registry, snapshot_callable=callback)
        jax.block_until_ready(final); jax.effects_barrier()
        np.testing.assert_allclose(writer.actual, times, rtol=0, atol=1.e-6)
        for path, expected in source_hashes.items():
            if sha(path) != expected:
                raise AssertionError(f'Integration source changed: {path}')
        meta.update(status='complete', actual_times=writer.actual, strip_sha256=writer.hashes)
    except Exception as error:
        meta.update(status='failed', error=str(error), actual_times=writer.actual, strip_sha256=writer.hashes)
        raise
    finally:
        meta_path.write_text(json.dumps(meta, indent=2)+'\n')
    from experiments.wind_bubble.review_collision_front_motion import run as review
    return review(args.output_dir, args.analysis_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-cells', type=int, default=128)
    parser.add_argument('--num-events', type=int, default=321)
    parser.add_argument('--t-end', type=float, default=.16)
    parser.add_argument('--reference-metadata', type=Path, default=REFERENCE)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--analysis-dir', type=Path)
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--analyze-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
