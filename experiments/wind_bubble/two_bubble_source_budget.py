"""Discrete EI source budgets for the fixed two-bubble experiment (NumPy only)."""
from __future__ import annotations
import numpy as np

SOURCES=((.35,.5,.5),(.65,.5,.5))
INJECTION_RADIUS=.0625


def injection_cell_counts(num_cells):
    if not isinstance(num_cells,(int,np.integer)) or num_cells<16 or num_cells%16:
        raise ValueError('Require num-cells to be a multiple of 16 and at least 16.')
    dx=1./num_cells;x=(np.arange(num_cells)+.5)*dx
    limit=(INJECTION_RADIUS-dx/2)**2
    counts=[int(((x[:,None,None]-sx)**2+(x[None,:,None]-sy)**2+(x[None,None,:]-sz)**2<=limit).sum())
            for sx,sy,sz in SOURCES]
    if not all(counts):raise ValueError('Empty injection mask.')
    return counts


def discrete_rates(num_cells,counts,mass_rate,velocity):
    if not np.isfinite([mass_rate,velocity]).all() or mass_rate<=0 or velocity<=0:
        raise ValueError('Require finite positive wind parameters.')
    fractions=np.asarray(counts,dtype=float)*(1./num_cells)**3/(4*np.pi*INJECTION_RADIUS**3/3)
    return fractions*mass_rate,fractions*(.5*velocity**2*mass_rate)


def budget_plan(num_cells,reference=None,mass_rate=.01,velocity=10.,actual_counts=None):
    counts=injection_cell_counts(num_cells)
    if actual_counts is not None and list(actual_counts)!=counts:
        raise ValueError('Solver geometry and preflight injection-mask counts disagree.')
    raw_mass,raw_energy=discrete_rates(num_cells,counts,mass_rate,velocity)
    scale=1.;target_mass=target_energy=None
    if reference is not None:
        if reference.get('status')!='complete':raise ValueError('Source-budget reference must be complete.')
        if not np.allclose(reference['source_positions'],SOURCES,rtol=0,atol=1e-12):raise ValueError('Source positions differ.')
        if not np.isclose(reference['injection_radius'],INJECTION_RADIUS,rtol=0,atol=1e-12):raise ValueError('Injection radius differs.')
        if reference.get('injection_prescription')!='existing EI analytic-volume normalization and radius-minus-half-cell mask':
            raise ValueError('Unsupported reference injection prescription.')
        if not np.isclose(reference['wind_terminal_velocity'],velocity,rtol=0,atol=1e-12):raise ValueError('Wind velocity differs.')
        ref_n=reference['num_cells'];ref_counts=injection_cell_counts(ref_n)
        if ref_counts!=reference['injection_cell_counts']:raise ValueError('Reference mask counts do not match its geometry.')
        expected_mass,expected_energy=discrete_rates(ref_n,ref_counts,reference['wind_mass_loss_rate_per_source'],velocity)
        target_mass=np.asarray(reference['discrete_mass_rates'],float)
        target_energy=np.asarray(reference['discrete_energy_rates'],float)
        for actual,expected in [(target_mass,expected_mass),(target_energy,expected_energy)]:
            if actual.shape!=(2,) or not np.isfinite(actual).all() or (actual<=0).any() or not np.allclose(actual,expected,rtol=1e-7,atol=1e-12):
                raise ValueError('Recorded reference rates are inconsistent with the EI budget.')
        scales=target_mass/raw_mass
        if not np.allclose(scales,scales[0],rtol=1e-10,atol=1e-12):
            raise ValueError('One shared source parameter cannot match both reference sources.')
        scale=float(scales[0])
    adjusted_mass=mass_rate*scale
    actual_mass,actual_energy=discrete_rates(num_cells,counts,adjusted_mass,velocity)
    if reference is not None:
        np.testing.assert_allclose(actual_mass,target_mass,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(actual_energy,target_energy,rtol=1e-7,atol=1e-12)
    return dict(num_cells=num_cells,injection_cell_counts=counts,parameter_scale=scale,
                energy_rate_definition='Integrated prescribed mechanical-energy source coefficient before the state-dependent EI pressure conversion; not measured net gas-energy deposition.',
                requested_mass_loss_rate_per_source=mass_rate,wind_mass_loss_rate_per_source=adjusted_mass,
                wind_terminal_velocity=velocity,nominal_wind_luminosity_per_source=.5*velocity**2*adjusted_mass,
                nominal_total_wind_luminosity=velocity**2*adjusted_mass,
                discrete_mass_rates=actual_mass.tolist(),discrete_energy_rates=actual_energy.tolist(),
                unmatched_discrete_mass_rates=raw_mass.tolist(),unmatched_discrete_energy_rates=raw_energy.tolist(),
                budget_matched=reference is not None,
                target_discrete_mass_rates=target_mass.tolist() if target_mass is not None else None,
                target_discrete_energy_rates=target_energy.tolist() if target_energy is not None else None)
