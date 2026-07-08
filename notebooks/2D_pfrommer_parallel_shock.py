# ============================================================================
# 2D Shock Finder Test — Two Parallel Rotated Shocks (blast wave in 1D)
# ============================================================================
# High pressure in the middle region, low pressure on both sides.
# This drives two shocks propagating outward in opposite directions
# along the shock normal — like a 1D blast wave, rotated by SHOCK_ANGLE.
#
# Initial conditions (three regions along the normal direction):
#   left region  (d < -1/6): rho=0.125, p=0.1   (low pressure)
#   mid  region  (-1/6<d<1/6): rho=1.0, p=1.0   (high pressure — the driver)
#   right region (d >  1/6): rho=0.125, p=0.1   (low pressure)
#
# Ground truth:
#   - two distinct shock surfaces, each perpendicular to the shock normal
#   - the shock normal has angle SHOCK_ANGLE from the x-axis
#   - shocks propagate in opposite directions → d_s points outward on each
#   - ds_x mean ≈ 0 (left shock cancels right shock), but per-shock ≈ ±cos θ
#   - Mach numbers should be roughly symmetric (same jump ratio on both sides)
# ============================================================================

#%%
from astrolink import AstroLink, visualize
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from astronomix import CARTESIAN, SimulationConfig, SimulationParams
from astronomix import get_helper_data, finalize_config
from astronomix import get_registered_variables, construct_primitive_state
from astronomix import time_integration
from astronomix.option_classes.simulation_config import HLLC, MINMOD
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix.option_classes.simulation_config import (
    GEOMETRY_TYPE,
    FIELD_TYPE,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

num_cells = 128
box_size  = 1.0

config = SimulationConfig(
    geometry=CARTESIAN,
    dimensionality=2,
    riemann_solver=HLLC,
    limiter=MINMOD,
    box_size=box_size,
    num_cells=num_cells,
)
params = SimulationParams(t_end=0.15)   # slightly shorter — keeps shocks separated

helper_data          = get_helper_data(config)
registered_variables = get_registered_variables(config)

# geometric_centers shape: (nx, ny, 2)  — last axis is (x, y)
geometric_centers = helper_data.geometric_centers

# helper_data.geometric_centers is a grid of nx × ny cells, where each cell contains its (x, y) coordinates.
geometry_x = geometric_centers[..., 0] # (nx, ny)
geometry_y = geometric_centers[..., 1] # (nx, ny)

# ============================================================================
# INITIAL CONDITIONS — double Sod (two outward-propagating shocks)
#
# A high-pressure driver region occupies the middle third of the domain
# along n̂ = (cos θ, sin θ) at θ = FRONT_NORMAL_ANGLE degrees:
#
#   left region  (d < -1/6): p=0.1, ρ=0.125  (ambient)
#   middle region(|d| < 1/6): p=1.0, ρ=1.0   (driver)
#   right region (d > +1/6): p=0.1, ρ=0.125  (ambient)
#
# The driver launches two shocks propagating in opposite directions along n̂.
# Both shocks should have the same Mach number and |normal| = FRONT_NORMAL_ANGLE.
# No initial velocity anywhere.
# ============================================================================
FRONT_NORMAL_ANGLE = 30.0        # angle of the vector perpendicular to the pressure discontinuity line
target_theta_rad = jnp.deg2rad(FRONT_NORMAL_ANGLE)
target_nx_hat    = jnp.cos(target_theta_rad)   # x-component of shock normal
target_ny_hat    = jnp.sin(target_theta_rad)   # y-component of shock normal

target_signed_dist = (geometry_x - 0.5) * target_nx_hat + (geometry_y - 0.5) * target_ny_hat

# three regions
in_left  = target_signed_dist < -1/6
in_right = target_signed_dist >  1/6
in_mid   = ~in_left & ~in_right

# high pressure driver in the middle → two shocks propagate outward
rho = jnp.where(in_mid, 1.0,   0.125)
p   = jnp.where(in_mid, 1.0,   0.1  )
u_x = jnp.zeros_like(geometry_x)
u_y = jnp.zeros_like(geometry_y)

initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho,
    velocity_x=u_x,
    velocity_y=u_y,
    gas_pressure=p,
)
config = finalize_config(config, initial_state.shape)

#%%
# RUN SIMULATION
final_state = time_integration(initial_state, config, params, registered_variables)
rho_final = final_state[registered_variables.density_index]
p_final   = final_state[registered_variables.pressure_index]


#%%
# RUN SHOCK FINDER

result = find_shocks_pfrommer(
    final_state,
    config,
    registered_variables,
    helper_data,
)
#%%
# RUN ASTROLINK ON DETECTED SHOCK-SURFACE CELLS

# Convert JAX arrays to NumPy.
geometry_x_np = np.array(geometry_x)
geometry_y_np = np.array(geometry_y)
surface_mask_np = np.array(result.shock_surface_cells, dtype=bool)

# Each detected shock-surface cell becomes one 2D point.
x_surface = geometry_x_np[surface_mask_np]
y_surface = geometry_y_np[surface_mask_np]

P = np.column_stack((x_surface, y_surface))

print("\n=== AstroLink shock clustering ===")
print("Shock-surface point-cloud shape:", P.shape)
if len(P) == 0:
    raise RuntimeError("No shock-surface points were detected.")

# Use physical x-y scaling without automatic feature rescaling.
clusterer = AstroLink(
    P,
    adaptive=0,
    verbose=1,
)

clusterer.run()

print("AstroLink cluster IDs:", clusterer.ids)
print("Number of hierarchy entries:", len(clusterer.clusters))
print("Cluster significances:", clusterer.significances)

# Plot AstroLink labels on the shock-surface point cloud.
visualize.labelsOnX(
    clusterer,
    P,
    skipZeroth=False,
)

plt.title("AstroLink clustering of two parallel shock surfaces")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

#%%
# PER-ENTITY SHOCK QUANTITIES

# Flatten the shock-surface values in the same order used to build P.
mach_surface = np.array(result.mach_numbers)[surface_mask_np]
flux_surface = np.array(result.thermal_energy_flux)[surface_mask_np]

shock_dir_x_surface = np.array(result.shock_direction[0])[surface_mask_np]
shock_dir_y_surface = np.array(result.shock_direction[1])[surface_mask_np]

for cluster_id, (start, end) in zip(
    clusterer.ids[1:],          # skip root cluster
    clusterer.clusters[1:],
):
    point_indices = clusterer.ordering[start:end]
    cluster_points = P[point_indices]

    centroid = cluster_points.mean(axis=0)

    cluster_mach = mach_surface[point_indices]
    cluster_flux = flux_surface[point_indices]

    cluster_dir_x = shock_dir_x_surface[point_indices]
    cluster_dir_y = shock_dir_y_surface[point_indices]

    mean_direction = np.array([
        cluster_dir_x.mean(),
        cluster_dir_y.mean(),
    ])

    direction_norm = np.linalg.norm(mean_direction)

    if direction_norm > 0:
        mean_direction /= direction_norm

    mean_angle_deg = np.degrees(
        np.arctan2(
            mean_direction[1],
            mean_direction[0],
        )
    )

    print(f"\nShock entity {cluster_id}")
    print(f"  number of surface cells : {len(point_indices)}")
    print(
        f"  centroid                : "
        f"({centroid[0]:.3f}, {centroid[1]:.3f})"
    )
    print(
        f"  Mach number             : "
        f"mean={cluster_mach.mean():.3f}, "
        f"min={cluster_mach.min():.3f}, "
        f"max={cluster_mach.max():.3f}"
    )
    print(
        f"  mean shock direction    : "
        f"({mean_direction[0]:.3f}, {mean_direction[1]:.3f})"
    )
    print(
        f"  mean direction angle    : "
        f"{mean_angle_deg:.2f} degrees"
    )
    print(
        f"  thermal-energy flux     : "
        f"mean={cluster_flux.mean():.6e}, "
        f"sum={cluster_flux.sum():.6e}"
    )
# %%
