"""Explicit source-strength setup for an unequal two-wind validation run."""
import numpy as np


def strength_plan(base_budget, strengths=None):
    values=np.ones(2) if strengths is None else np.asarray(strengths,dtype=float)
    if values.shape!=(2,) or not np.isfinite(values).all() or (values<=0).any():
        raise ValueError('Require two finite positive source strengths.')
    mass=float(base_budget['wind_mass_loss_rate_per_source'])
    velocity=float(base_budget['wind_terminal_velocity'])
    return dict(source_strengths=values.tolist(),source_mass_loss_rates=(mass*values).tolist(),
        nominal_wind_luminosities=(.5*velocity**2*mass*values).tolist(),
        discrete_mass_rates=(np.asarray(base_budget['discrete_mass_rates'])*values).tolist(),
        discrete_energy_rates=(np.asarray(base_budget['discrete_energy_rates'])*values).tolist(),
        equal_source_reference_preserved=bool(np.all(values==1)))
