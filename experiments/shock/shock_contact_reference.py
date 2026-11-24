"""Independent exact Euler fan containing a contact followed by a shock.

The left acoustic wave has zero strength. Pressure and velocity match across
the contact; the right-going shock obeys the entropy-satisfying jump conditions.
All states are built in NumPy, without importing the production shock finder.
Conserved cell averages are exact for this piecewise-constant solution.
"""
from dataclasses import dataclass

import numpy as np

from experiments.shock.colliding_shocks_reference import (
    GAMMA, box_filter, conserved, flux, primitive, shock_ratios,
)


@dataclass(frozen=True)
class ShockContactReference:
    mach: float = 3.0
    contact_density_factor: float = 8.0
    gamma: float = GAMMA

    def __post_init__(self):
        shock_ratios(self.mach, self.gamma)
        if self.gamma != GAMMA:
            raise ValueError('Use gamma=5/3 to match the production finder.')
        if not np.isfinite(self.contact_density_factor) or self.contact_density_factor <= 0:
            raise ValueError('Contact density factor must be finite and positive.')

    @property
    def upstream(self):
        return np.array([1., 0., 1.])

    @property
    def shock_speed(self):
        return self.mach * np.sqrt(self.gamma)

    @property
    def downstream(self):
        ratio, pressure = shock_ratios(self.mach, self.gamma)
        return np.array([ratio, self.shock_speed * (1 - 1 / ratio), pressure])

    @property
    def contact_speed(self):
        return self.downstream[1]

    @property
    def contact_gas(self):
        state = self.downstream.copy()
        state[0] *= self.contact_density_factor
        return state

    @property
    def states(self):
        return np.stack([self.contact_gas, self.downstream, self.upstream])

    def positions(self, time, x0=.25):
        if not np.isfinite(time) or time < 0 or not np.isfinite(x0):
            raise ValueError('Require a finite nonnegative time and finite origin.')
        return np.array([x0 + self.contact_speed * time, x0 + self.shock_speed * time])

    def sample(self, x, time, x0=.25):
        contact, shock = self.positions(time, x0)
        return self.states[np.where(np.asarray(x) < contact, 0,
                                    np.where(np.asarray(x) < shock, 1, 2))]

    def profile(self, resolution, separation_cells, filter_cells=1, phase=0., orientation=1):
        """Place a known fan at a prescribed grid separation and shock phase.

        Shock phase 0 places the shock on a cell edge; phase 0.5 places it at
        a cell center. The underlying Riemann origin can be outside this
        observation window: no finite-domain evolution or boundary condition
        is asserted. Larger box filters are synthetic broadening controls.
        """
        if resolution < 16 or resolution % 2:
            raise ValueError('Use an even grid with at least 16 cells.')
        if not np.isfinite(separation_cells) or separation_cells <= 0:
            raise ValueError('Separation must be finite and positive.')
        if not np.isfinite(phase) or not 0 <= phase < 1:
            raise ValueError('Shock phase must lie in [0,1).')
        if orientation not in (-1, 1):
            raise ValueError('Orientation must be +1 or -1.')
        gap = separation_cells / resolution
        shock = .5 + phase / resolution
        contact = shock - gap
        if not 0 < contact < shock < 1:
            raise ValueError('Both waves must lie inside the observation window.')
        time = gap / (self.shock_speed - self.contact_speed)
        x0 = shock - self.shock_speed * time
        profile = self.cell_averages(resolution, time, x0, filter_cells)
        if orientation == -1:
            profile = profile[::-1].copy()
            profile[:, 1] *= -1
            contact, shock = 1 - contact, 1 - shock
        return profile, dict(contact_x=contact, shock_x=shock, mach=self.mach,
            contact_density_factor=self.contact_density_factor, orientation=orientation,
            time=time, riemann_origin=x0, has_contact=self.contact_density_factor != 1.,
            shock_speed=orientation * self.shock_speed,
            contact_speed=orientation * self.contact_speed)

    def cell_averages(self, resolution, time, x0=.25, filter_cells=1):
        contact, shock = self.positions(time, x0)
        edges = np.arange(resolution + 1) / resolution
        a = np.clip((contact - edges[:-1]) * resolution, 0, 1)
        c = np.clip((edges[1:] - shock) * resolution, 0, 1)
        weights = np.stack([a, np.maximum(1 - a - c, 0), c], axis=1)
        average = weights @ conserved(self.states, self.gamma)
        return primitive(box_filter(average, filter_cells), self.gamma)

    def control(self, resolution, kind, filter_cells=1, phase=0., orientation=1):
        if kind == 'shock':
            states = np.stack([self.downstream, self.upstream])
        elif kind == 'contact':
            states = np.stack([self.contact_gas, self.downstream])
        else:
            raise ValueError('Control kind must be shock or contact.')
        position = .5 + phase / resolution
        edges = np.arange(resolution + 1) / resolution
        a = np.clip((position - edges[:-1]) * resolution, 0, 1)
        average = np.stack([a, 1 - a], axis=1) @ conserved(states, self.gamma)
        profile = primitive(box_filter(average, filter_cells), self.gamma)
        if orientation == -1:
            profile = profile[::-1].copy()
            profile[:, 1] *= -1
            position = 1 - position
        return profile, dict(contact_x=position if kind == 'contact' else None,
            shock_x=position if kind == 'shock' else None, mach=self.mach,
            contact_density_factor=self.contact_density_factor, orientation=orientation,
            time=None, riemann_origin=None, has_contact=kind == 'contact',
            shock_speed=orientation * self.shock_speed,
            contact_speed=orientation * self.contact_speed)

    def flux_residuals(self):
        """Check all three conservation laws at the shock and contact."""
        results = []
        for left, right, speed in (
            (self.contact_gas, self.downstream, self.contact_speed),
            (self.downstream, self.upstream, self.shock_speed),
        ):
            a, b = conserved(left, self.gamma), conserved(right, self.gamma)
            fa, fb = flux(left, self.gamma), flux(right, self.gamma)
            scale = np.maximum.reduce([abs(fa), abs(fb), abs(speed*a), abs(speed*b), np.ones(3)])
            results.append((fb - fa - speed*(b - a)) / scale)
        return np.asarray(results)
