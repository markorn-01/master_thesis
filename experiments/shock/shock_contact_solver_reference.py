"""Wind-unit adapter for the established exact shock/contact reference.

Both interfaces satisfy the Euler jump conditions. Their common downstream
pressure/velocity leave only a contact on the left and a shock on the right.
No source, rarefaction, pressure-based tracking, or numerical solver is used
to construct this reference. Finite-volume averages are exact in conserved
variables, including cells intersecting both interfaces.
"""
from dataclasses import dataclass

import numpy as np

from experiments.shock.colliding_shocks_reference import conserved, primitive, flux, shock_ratios
from experiments.shock.shock_contact_reference import ShockContactReference as UnitReference


@dataclass(frozen=True)
class ShockContactReference:
    mach: float = 8.
    upstream_density: float = 1.
    upstream_pressure: float = .01
    contact_density: float = .04
    origin: float = .625
    gamma: float = 5/3

    def __post_init__(self):
        if (self.gamma != 5/3 or not np.isfinite([self.mach, self.upstream_density,
                self.upstream_pressure, self.contact_density, self.origin]).all()
                or min(self.upstream_density, self.upstream_pressure, self.contact_density) <= 0
                or not 0 < self.origin < 1):
            raise ValueError('Require positive finite states, gamma=5/3, and an interior origin.')
        shock_ratios(self.mach, self.gamma)

    @property
    def unit_reference(self):
        density, _ = shock_ratios(self.mach, self.gamma)
        return UnitReference(self.mach, self.contact_density/(density*self.upstream_density), self.gamma)

    @property
    def velocity_scale(self):
        return np.sqrt(self.upstream_pressure/self.upstream_density)

    def scale(self, q):
        return np.asarray(q)*[self.upstream_density, self.velocity_scale, self.upstream_pressure]

    @property
    def shock_speed(self):
        return self.unit_reference.shock_speed*self.velocity_scale

    @property
    def upstream(self):
        return self.scale(self.unit_reference.upstream)

    @property
    def shocked(self):
        return self.scale(self.unit_reference.downstream)

    @property
    def contact_state(self):
        return self.scale(self.unit_reference.contact_gas)

    def positions(self, time):
        if not np.isfinite(time) or time < 0:
            raise ValueError('Require a finite nonnegative time.')
        contact, shock = self.unit_reference.positions(time*self.velocity_scale, self.origin)
        if shock >= 1:
            raise ValueError('Reference shock has reached the domain boundary.')
        return float(contact), float(shock)

    def cell_averages(self, n, time, contact_only=False):
        if not isinstance(n, (int, np.integer)) or n < 16:
            raise ValueError('Require an integer resolution of at least 16.')
        contact, shock = self.positions(time)
        edges = np.arange(n+1)/n
        left = np.clip((contact-edges[:-1])*n, 0, 1)
        if contact_only:
            state = left[:, None]*conserved(self.contact_state)+(1-left[:, None])*conserved(self.shocked)
        else:
            return self.scale(self.unit_reference.cell_averages(n, time*self.velocity_scale, self.origin))
        return primitive(state)

    def point_values(self, x, time, contact_only=False):
        contact, shock = self.positions(time)
        if not contact_only:
            return self.scale(self.unit_reference.sample(x, time*self.velocity_scale, self.origin))
        q = np.tile(self.shocked, (len(x), 1))
        q[np.asarray(x) < contact] = self.contact_state
        return q

    def flux_residuals(self):
        rows = []
        for left, right, speed in ((self.contact_state, self.shocked, self.shocked[1]),
                                   (self.shocked, self.upstream, self.shock_speed)):
            a, b = conserved(left), conserved(right)
            fa, fb = flux(left), flux(right)
            scale = np.maximum.reduce([abs(fa), abs(fb), abs(speed*a), abs(speed*b), np.ones(3)])
            rows.append((fb-fa-speed*(b-a))/scale)
        return np.array(rows)
