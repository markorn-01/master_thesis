"""Fixed non-collinear three-wind setup and explicit per-source budget matching."""
import numpy as np
from experiments.wind_bubble.two_bubble_source_budget import INJECTION_RADIUS,discrete_rates,budget_plan

# Binary-exact coordinates and reflected/permuted offsets give identical mask
# counts, allowing one shared mass parameter without hidden per-source scaling.
SOURCES=((.375,.375,.5),(.625,.375,.5),(.5,.625,.375))


def injection_counts(num_cells):
    if not isinstance(num_cells,(int,np.integer)) or num_cells<16 or num_cells%16:
        raise ValueError('Require a multiple of 16 cells per axis.')
    x=(np.arange(num_cells)+.5)/num_cells;limit=(INJECTION_RADIUS-.5/num_cells)**2
    counts=[int(((x[:,None,None]-s[0])**2+(x[None,:,None]-s[1])**2+(x[None,None,:]-s[2])**2<=limit).sum()) for s in SOURCES]
    if not all(counts) or len(set(counts))!=1:raise ValueError('Require three nonempty equal-sized injection masks.')
    return counts


def source_budget(num_cells,reference=None,mass_rate=.01,velocity=10.,actual_counts=None):
    counts=injection_counts(num_cells)
    if actual_counts is not None and list(actual_counts)!=counts:raise ValueError('Actual source masks differ from preflight.')
    raw_mass,raw_energy=discrete_rates(num_cells,counts,mass_rate,velocity)
    scale=1.;target=None
    if reference is not None:
        # Reuse strict validation of the OLD equal two-source reference itself.
        checked=budget_plan(reference['num_cells'],reference,
            reference['wind_mass_loss_rate_per_source'],velocity)
        if not np.allclose(checked['discrete_mass_rates'],checked['discrete_mass_rates'][0],rtol=1e-12,atol=0):
            raise ValueError('Require an equal-strength source budget reference.')
        target=float(checked['discrete_mass_rates'][0]);scale=target/raw_mass[0]
    adjusted=mass_rate*scale;mass,energy=discrete_rates(num_cells,counts,adjusted,velocity)
    if target is not None:np.testing.assert_allclose(mass,target,rtol=1e-12,atol=1e-12)
    return dict(num_cells=num_cells,injection_cell_counts=counts,parameter_scale=float(scale),
        requested_mass_loss_rate_per_source=mass_rate,wind_mass_loss_rate_per_source=float(adjusted),wind_terminal_velocity=velocity,
        nominal_wind_luminosity_per_source=.5*velocity**2*adjusted,nominal_total_wind_luminosity=3*.5*velocity**2*adjusted,
        discrete_mass_rates=mass.tolist(),discrete_energy_rates=energy.tolist(),budget_matched=reference is not None,
        energy_rate_definition='Prescribed mechanical source coefficient before state-dependent EI pressure conversion; not net gas-energy deposition.',
        reference_scope='Each of three sources matches one source of the equal two-wind reference; total prescribed budget is 1.5 times its total.')


def weighted_budget(base,strengths=None):
    values=np.ones(3) if strengths is None else np.asarray(strengths,float)
    if values.shape!=(3,) or not np.isfinite(values).all() or (values<=0).any():raise ValueError('Require three positive source strengths.')
    mass=base['wind_mass_loss_rate_per_source'];v=base['wind_terminal_velocity']
    return dict(source_strengths=values.tolist(),source_mass_loss_rates=(mass*values).tolist(),
        nominal_wind_luminosities=(.5*v*v*mass*values).tolist(),
        discrete_mass_rates=(np.asarray(base['discrete_mass_rates'])*values).tolist(),
        discrete_energy_rates=(np.asarray(base['discrete_energy_rates'])*values).tolist(),
        equal_source_reference_preserved=False,
        equal_source_strengths=bool(np.all(values==1)),
        per_source_reference_budget_preserved=bool(base['budget_matched'] and np.all(values==1)))
