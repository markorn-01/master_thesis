"""Explain the eight aligned central-slice candidates in the 128^3 t=0.12 replay.

Read-only decomposition of the existing extension, not another detector or
physical shock classifier. The primary saved-state comparison is unchanged.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from astronomix import (CARTESIAN, SimulationConfig, construct_primitive_state,
    finalize_config, get_registered_variables)
from astronomix.option_classes.simulation_config import StaticFloatVector
from astronomix._physics_modules._shock_finder._gradients import (
    _calculate_shock_direction, _calculate_temperature_gradient,
    _calculate_density_gradient, _calculate_velocity_divergence)
from astronomix._physics_modules._shock_finder._shock_zones import (
    _make_interior_mask, get_adaptive_post_pre_shock_values,
    _sample_field_at_distances, _transition_jump_candidates)
from experiments.shock.run_evolved_collision import sha
from experiments.wind_bubble.compare_collision_jump_screening import (
    ROOTS, CENTRAL, HALO, crop_layout, native_axis_indices)


def run(directory):
    directory = Path(directory); n, event = 128, 12
    path = directory/'verification.json'; primary = json.loads(path.read_text())
    inputs = {str(path):sha(path)}
    for name, digest in {**primary['source_sha256'], **primary['input_sha256']}.items():
        if sha(name) != digest:
            raise AssertionError(f'Primary replay source or input changed: {name}')
    state_path = ROOTS[1]/'audit'/f'state_{event:04d}.npy'
    comparison_path = directory/f'comparison_n{n:03d}_{event:04d}.npz'
    inputs.update({str(state_path):sha(state_path), str(comparison_path):sha(comparison_path)})
    with np.load(comparison_path, allow_pickle=False) as data:
        saved = {key:data[key] for key in data.files}
    centers = saved['centers']; _, k = native_axis_indices(centers, n)
    roi = ((centers >= CENTRAL[:,0]) & (centers <= CENTRAL[:,1])).all(axis=-1)
    plane = np.zeros(roi.shape, bool); plane[:,:,k] = True
    selected = roi & plane & saved['compression'] & saved['gradient_alignment'] & ~saved['transition_zone']
    crop, target, _ = crop_layout(n, HALO); shape = tuple(s.stop-s.start for s in crop)
    config = SimulationConfig(geometry=CARTESIAN, dimensionality=3, num_cells=shape[0],
        box_size=StaticFloatVector(*[size/n for size in shape]), mhd=False)
    registry = get_registered_variables(config); one, zero = jnp.ones(shape), jnp.zeros(shape)
    template = construct_primitive_state(config=config, registered_variables=registry,
        density=one, gas_pressure=one, velocity_x=zero, velocity_y=zero, velocity_z=zero)
    config = finalize_config(config, template.shape)

    @jax.jit
    def inspect(state):
        p, rho = state[registry.pressure_index], state[registry.density_index]
        direction = _calculate_shock_direction(p, rho, config)
        comp = _calculate_velocity_divergence(state, config, registry) < 0
        aligned = jnp.sum(_calculate_temperature_gradient(p, rho, config)
                         * _calculate_density_gradient(rho, config), axis=0) > 0
        provisional = comp & aligned & _make_interior_mask(shape)
        p2,p1,r2,r1,valid,post,pre = get_adaptive_post_pre_shock_values(
            direction, provisional, p, rho, max_steps=8)
        s = jnp.log(p)-(5/3)*jnp.log(rho)
        s2,s1 = _sample_field_at_distances(s, direction, post, pre)
        lp,lr = jnp.log(p2)-jnp.log(p1), jnp.log(r2)-jnp.log(r1)
        pmin = (2*(5/3)*1.3**2-2/3)/(8/3)
        rmin = (8/3)*1.3**2/((2/3)*1.3**2+2)
        fields = dict(valid=valid, pressure_ratio=p2/p1, temperature_ratio=(p2/r2)/(p1/r1),
            entropy_jump=s2-s1, pressure_pass=lp >= jnp.log(pmin),
            temperature_pass=lp-lr >= jnp.log(pmin/rmin), entropy_pass=s2-s1 > 1e-5,
            production_extra=_transition_jump_candidates(p, rho, direction, provisional))
        return {name:value[target] for name,value in fields.items()}

    raw = np.load(state_path, allow_pickle=False)
    fields = {name:np.asarray(value) for name,value in inspect(jnp.asarray(raw[(slice(None),)+crop])).items()}
    np.testing.assert_array_equal(fields['production_extra'][selected], saved['extra_jump'][selected])
    rows = []
    for index in np.argwhere(selected):
        cell = tuple(index)
        rows.append(dict(zip('xyz', map(float, centers[cell]))))
        rows[-1].update({name:values[cell].item() for name,values in fields.items()})
    pmin = (2*(5/3)*1.3**2-2/3)/(8/3)
    rmin = (8/3)*1.3**2/((2/3)*1.3**2+2)
    sources = [Path(__file__), Path('experiments/wind_bubble/compare_collision_jump_screening.py')]
    sources += sorted(Path('astronomix/_physics_modules/_shock_finder').glob('*.py'))
    result = dict(review_completed=True, resolution=n, event=event, candidate_cells=len(rows),
        valid_samples=sum(r['valid'] for r in rows), pressure_pass=sum(r['pressure_pass'] for r in rows),
        temperature_pass=sum(r['temperature_pass'] for r in rows), entropy_pass=sum(r['entropy_pass'] for r in rows),
        required_pressure_ratio=pmin, required_temperature_ratio=pmin/rmin,
        maximum_sampled_pressure_ratio=max((r['pressure_ratio'] for r in rows), default=None),
        maximum_sampled_temperature_ratio=max((r['temperature_ratio'] for r in rows), default=None),
        rows=rows, input_sha256=inputs, source_sha256={str(p):sha(p) for p in sources},
        scope='Decomposition of eight candidate-cell rejections, not proof of physical shock strength or absence.')
    for name,digest in inputs.items():
        if sha(name) != digest:
            raise AssertionError(f'Review input changed: {name}')
    output = directory/'extension_exception.json'
    if output.exists():
        raise ValueError('Do not overwrite an existing review.')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','input_sha256','source_sha256')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir', type=Path,
        default=Path('outputs/bubble_transition_jump_comparison_n064_n128'))
    run(parser.parse_args().comparison_dir)
