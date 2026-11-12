"""
Configuration and parameter containers for stellar-wind injection.

``WindConfig`` holds the static choices (which injection scheme, how many cells
to inject into) and ``WindParams`` the physical wind parameters. The module-level
integer constants name the available injection schemes from
https://arxiv.org/abs/2107.14673.
"""

# typing
from typing import NamedTuple

# Wind injection schemes (see https://arxiv.org/abs/2107.14673).
MEO = 0  # momentum and energy overwrite
EI = 1  # thermal energy injection
MEI = 2  # momentum and energy injection


class WindConfig(NamedTuple):
    stellar_wind: bool = False
    num_injection_cells: int = 10
    wind_injection_scheme: int = EI
    trace_wind_density: bool = False
    # Optional stationary sources in physical coordinates. Empty keeps the
    # original centered source. Currently supported for 3D Cartesian FV EI.
    # WindParams describes EACH source, rather than their combined luminosity.
    source_positions: tuple[tuple[float, float, float], ...] = ()
    # Optional positive per-source mass and mechanical-luminosity multipliers.
    # Empty preserves identical-source behavior; currently 3D FV EI only.
    source_strengths: tuple[float, ...] = ()


class WindParams(NamedTuple):
    wind_mass_loss_rate: float = 0.0
    wind_final_velocity: float = 0.0

    # Only required for the MEO injection scheme.
    pressure_floor: float = 100000.0
