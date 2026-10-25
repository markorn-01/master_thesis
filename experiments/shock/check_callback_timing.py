"""Audit existing snapshot callbacks using small uniform hydrodynamic runs.

This reports current behavior; it does not modify the time integrator or claim
that all reported readiness checks pass. The early explicit-time probe now uses adaptive steps, after the scheduling
correction. Historical pre-correction observations remain in
outputs/callback_timing_audit/.

Run: python -m experiments.shock.check_callback_timing
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from astronomix import (
    CARTESIAN, SimulationConfig, SimulationParams, construct_primitive_state,
    finalize_config, get_registered_variables, time_integration,
)
from astronomix.option_classes.simulation_config import SnapshotSettings


def run_probe(name, *, requested_times=None, stored=False, fixed=False, t_start=0.0, pressure=1.0e-4):
    """Exercise the public integration API with real, inexpensive gas updates."""
    t_end = t_start + 0.5
    config = SimulationConfig(
        geometry=CARTESIAN, dimensionality=1, num_cells=16, box_size=1.0,
        activate_snapshot_callback=True, return_snapshots=stored,
        snapshot_settings=SnapshotSettings(return_states=True, return_final_state=True),
        num_snapshots=4, use_specific_snapshot_timepoints=requested_times is not None,
        fixed_timestep=fixed, num_timesteps=20,
        print_elapsed_time=False, progress_bar=False,
    )
    variables = get_registered_variables(config)
    state = construct_primitive_state(
        config=config, registered_variables=variables,
        density=jnp.ones(16), velocity_x=jnp.zeros(16),
        gas_pressure=jnp.full(16, pressure),
    )
    config = finalize_config(config, state.shape)
    params = SimulationParams(
        t_start=t_start, t_end=t_end,
        snapshot_timepoints=jnp.asarray(requested_times if requested_times is not None else [0.0]),
    )
    records = []

    def collect(time, shape, rho_mean):
        records.append(dict(time=float(time), shape=np.asarray(shape).tolist(),
                            density_mean=float(rho_mean)))

    def callback(time, primitive, registry):
        # Only scalars and shape metadata cross to the host, not full states.
        jax.debug.callback(collect, time, jnp.asarray(primitive.shape),
                           jnp.mean(primitive[registry.density_index]), ordered=True)

    result = time_integration(state, config, params, variables, snapshot_callable=callback)
    jax.block_until_ready(result)
    jax.effects_barrier()
    actual = np.asarray([r["time"] for r in records])
    # Regular callback interpretation: four intervals, plus the final state.
    expected = np.asarray(requested_times if requested_times is not None
                          else np.linspace(t_start, t_end, 5))
    if expected[-1] < t_end - 2e-6:
        expected = np.append(expected, t_end)
    matches = actual.shape == expected.shape and np.allclose(actual, expected, rtol=0, atol=2e-6)
    # A finite uniform solution is required even when a scheduling check fails.
    final = np.asarray(result.final_state if stored else result)
    if not np.isfinite(final).all():
        raise AssertionError(f"{name}: non-finite hydrodynamic result")
    np.testing.assert_allclose(final[variables.density_index], 1.0, atol=1e-6)
    report = dict(
        case=name, backend="fixed_step" if fixed else "adaptive_while",
        requested_times=expected.tolist(), observed_callbacks=records,
        callback_times_match=bool(matches),
        callback_final_at_t_end=bool(len(actual) and abs(actual[-1]-t_end) <= 2e-6),
        callback_state_is_unpadded=bool(records and all(r["shape"] == list(state.shape) for r in records)),
        final_state_shape=list(final.shape),
    )
    if stored:
        report["stored_times"] = np.asarray(result.time_points).tolist()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/callback_timing_validation"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probes = [
        ("regular_adaptive_large_step", {}),
        ("regular_adaptive_smaller_steps", dict(pressure=0.01)),
        ("specific_late_adaptive", dict(requested_times=[0.0, 0.20, 0.35, 0.50])),
        ("specific_early_adaptive", dict(requested_times=[0.0, 0.05, 0.31, 0.50])),
        ("specific_without_initial", dict(requested_times=[0.1, 0.2, 0.35, 0.5])),
        ("specific_without_final", dict(requested_times=[0.0, 0.05, 0.31])),
        ("regular_fixed_aligned", dict(fixed=True)),
        ("stored_and_callback", dict(requested_times=[0.0, 0.20, 0.35, 0.50], stored=True)),
        ("regular_nonzero_start", dict(t_start=0.20)),
    ]
    cases = []
    for name, kwargs in probes:
        report = run_probe(name, **kwargs)
        cases.append(report)
        print(f"{name}: callbacks={[r['time'] for r in report['observed_callbacks']]}; "
              f"times_match={report['callback_times_match']}; "
              f"unpadded={report['callback_state_is_unpadded']}", flush=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    payload = dict(
        purpose="Behavior audit, not a passing regression-suite declaration",
        commit=revision,
        integrator_sha256=hashlib.sha256(
            Path("astronomix/time_stepping/time_integration.py").read_bytes()
        ).hexdigest(), python=platform.python_version(), jax=jax.__version__,
        device_backend=jax.default_backend(),
        regular_schedule_convention="num_snapshots intervals plus a final callback; proposed exact-time contract",
        scope="Post-correction timing checks; padding and stored-snapshot precedence are retained",
        cases=cases,
    )
    path = args.output_dir / "callback_timing.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Audit written to {path}", flush=True)


if __name__ == "__main__":
    main()
