# ============================================================================
# 2D Shock Finder Test — Point Explosion (Sedov-like, single outward shock)
# ============================================================================
# A single point-like energy injection at the domain center drives one
# outward-propagating circular shock (Sedov-Taylor-like blast wave).
#
# Initial conditions:
#   A small disk of radius r_explosion at the center is set to a high
#   pressure p_explosion_gas, chosen so that integrating p/(gamma-1) over
#   the disk area gives back the target explosion energy E_explosion.
#   Everywhere else: ambient density/pressure, no initial velocity.
#
# Stress test goals (NOT a clean Sedov validation):
#   1. A single closed circular shock surface is detected
#   2. Shock finder does not crash or produce garbage everywhere
#   3. Detected shock_direction points radially outward from the center
#      along the shock front, at all angles (azimuthal symmetry check)
#   4. The very center (~r < r_explosion, pre-shock-formation region) and
#      the exact center point itself are expected to be ambiguous/noisy
# ============================================================================

#%%
from astrolink import AstroLink, visualize
from scipy.ndimage import label
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from astronomix import CARTESIAN, SimulationConfig, SimulationParams
from astronomix import get_helper_data, finalize_config
from astronomix import get_registered_variables, construct_primitive_state
from astronomix import time_integration
from astronomix.option_classes.simulation_config import HLLC, MINMOD
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

#%%
# CONFIGURATION

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
params = SimulationParams(t_end=0.15)

helper_data          = get_helper_data(config)
registered_variables = get_registered_variables(config)

# geometric_centers shape: (nx, ny, 2)  — last axis is (x, y)
geometric_centers = helper_data.geometric_centers

# helper_data.geometric_centers is a grid of nx × ny cells, where each cell contains its (x, y) coordinates.
geometry_x = geometric_centers[..., 0]  # (nx, ny)
geometry_y = geometric_centers[..., 1]  # (nx, ny)


# ============================================================================
# INITIAL CONDITIONS — point explosion (single outward-propagating shock)
#
# A small disk of radius r_explosion centered at TARGET_CENTER is given a
# uniform high pressure such that integrating p/(gamma-1) over the disk
# area reproduces E_explosion. Outside the disk: ambient density/pressure.
# No initial velocity anywhere.
# The center of the domain is the region of interest (single point source).
# ============================================================================

# center of the explosion
TARGET_CENTER = (0.5, 0.5)
center_x, center_y = TARGET_CENTER

# total explosion energy (code units)
E_explosion = 1.0

# ambient (background) physical conditions
rho_ambient = 1.0
p_ambient   = 1e-4

# radius of the injection disk (code units)
r_explosion = 0.05

# distance of every cell from the explosion center
dx_from_center = geometry_x - center_x
dy_from_center = geometry_y - center_y

r = jnp.sqrt(dx_from_center**2 + dy_from_center**2)

# injection area (2D analog of the 3D injection_volume in the point-explosion setup)
injection_area = jnp.pi * r_explosion**2

# adiabatic index of the gas
gamma_gas = params.gamma

# E = p * A / (gamma - 1)  =>  p = E * (gamma - 1) / A
p_explosion_gas = E_explosion * (gamma_gas - 1) / injection_area

# pressure: high within the explosion disk, ambient elsewhere
p   = jnp.where(r < r_explosion, p_explosion_gas, p_ambient)
rho = jnp.ones_like(geometry_x) * rho_ambient
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


# ============================================================================
# RUN SIMULATION
# ============================================================================

#%%
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
# geometry_x_np = np.array(geometry_x)
# geometry_y_np = np.array(geometry_y)
# surface_mask_np = np.array(result.shock_surface_cells, dtype=bool)

# # Each detected shock-surface cell becomes one 2D point.
# x_surface = geometry_x_np[surface_mask_np]
# y_surface = geometry_y_np[surface_mask_np]

# P = np.column_stack((x_surface, y_surface))

# print("\n=== AstroLink shock clustering ===")
# print("Shock-surface point-cloud shape:", P.shape)
# if len(P) == 0:
#     raise RuntimeError("No shock-surface points were detected.")

# # Use physical x-y scaling without automatic feature rescaling.
# clusterer = AstroLink(
#     P,
#     adaptive=0,
#     d_intrinsic=1,
#     verbose=1,
# )

# clusterer.run()

# print("AstroLink cluster IDs:", clusterer.ids)
# print("Number of hierarchy entries:", len(clusterer.clusters))
# print("Cluster significances:", clusterer.significances)

# # Plot AstroLink labels on the shock-surface point cloud.
# visualize.labelsOnX(
#     clusterer,
#     P,
#     skipZeroth=False,
# )

# plt.title("AstroLink hierarchy for the Sedov shock surface")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.axis("equal")
# plt.show()

# #%%
# # PER-ENTITY SHOCK QUANTITIES

# # Flatten the shock-surface values in the same order used to build P.
# mach_surface = np.array(result.mach_numbers)[surface_mask_np]
# flux_surface = np.array(result.thermal_energy_flux)[surface_mask_np]

# shock_dir_x_surface = np.array(result.shock_direction[0])[surface_mask_np]
# shock_dir_y_surface = np.array(result.shock_direction[1])[surface_mask_np]

# for cluster_id, (start, end) in zip(
#     clusterer.ids[1:],          # skip root cluster
#     clusterer.clusters[1:],
# ):
#     point_indices = clusterer.ordering[start:end]
#     cluster_points = P[point_indices]

#     centroid = cluster_points.mean(axis=0)

#     cluster_mach = mach_surface[point_indices]
#     cluster_flux = flux_surface[point_indices]

#     cluster_dir_x = shock_dir_x_surface[point_indices]
#     cluster_dir_y = shock_dir_y_surface[point_indices]

#     mean_direction = np.array([
#         cluster_dir_x.mean(),
#         cluster_dir_y.mean(),
#     ])

#     direction_norm = np.linalg.norm(mean_direction)

#     if direction_norm > 0:
#         mean_direction /= direction_norm

#     mean_angle_deg = np.degrees(
#         np.arctan2(
#             mean_direction[1],
#             mean_direction[0],
#         )
#     )

#     print(f"\nShock entity {cluster_id}")
#     print(f"  number of surface cells : {len(point_indices)}")
#     print(
#         f"  centroid                : "
#         f"({centroid[0]:.3f}, {centroid[1]:.3f})"
#     )
#     print(
#         f"  Mach number             : "
#         f"mean={cluster_mach.mean():.3f}, "
#         f"min={cluster_mach.min():.3f}, "
#         f"max={cluster_mach.max():.3f}"
#     )
#     print(
#         f"  mean shock direction    : "
#         f"({mean_direction[0]:.3f}, {mean_direction[1]:.3f})"
#     )
#     print(
#         f"  mean direction angle    : "
#         f"{mean_angle_deg:.2f} degrees"
#     )
#     print(
#         f"  thermal-energy flux     : "
#         f"mean={cluster_flux.mean():.6e}, "
#         f"sum={cluster_flux.sum():.6e}"
#     )

# %%

#%%
# IDENTIFY SPATIALLY CONNECTED SHOCK ENTITIES

surface_mask_np = np.array(
    result.shock_surface_cells,
    dtype=bool,
)

# 8-neighbour connectivity in 2D:
# horizontal, vertical, and diagonal neighbours are connected.
connectivity = np.ones((3, 3), dtype=np.int8)

component_labels, num_components = label(
    surface_mask_np,
    structure=connectivity,
)

print("\n=== Connected shock entities ===")
print("Number of connected components:", num_components)

geometry_x_np = np.array(geometry_x)
geometry_y_np = np.array(geometry_y)

mach_np = np.array(result.mach_numbers)
flux_np = np.array(result.thermal_energy_flux)
dir_x_np = np.array(result.shock_direction[0])
dir_y_np = np.array(result.shock_direction[1])

for entity_id in range(1, num_components + 1):
    entity_mask = component_labels == entity_id

    x_entity = geometry_x_np[entity_mask]
    y_entity = geometry_y_np[entity_mask]

    mach_entity = mach_np[entity_mask]
    flux_entity = flux_np[entity_mask]
    dir_x_entity = dir_x_np[entity_mask]
    dir_y_entity = dir_y_np[entity_mask]

    centroid = np.array([
        x_entity.mean(),
        y_entity.mean(),
    ])

    mean_direction = np.array([
        dir_x_entity.mean(),
        dir_y_entity.mean(),
    ])

    direction_norm = np.linalg.norm(mean_direction)

    if direction_norm > 0:
        mean_direction /= direction_norm

    direction_angle = np.degrees(
        np.arctan2(
            mean_direction[1],
            mean_direction[0],
        )
    )

    print(f"\nShock entity {entity_id}")
    print(f"  surface cells           : {entity_mask.sum()}")
    print(
        f"  centroid                : "
        f"({centroid[0]:.3f}, {centroid[1]:.3f})"
    )
    print(
        f"  Mach number             : "
        f"mean={mach_entity.mean():.3f}, "
        f"min={mach_entity.min():.3f}, "
        f"max={mach_entity.max():.3f}"
    )
    print(
        f"  mean shock direction    : "
        f"({mean_direction[0]:.3f}, {mean_direction[1]:.3f})"
    )
    print(
        f"  direction angle         : "
        f"{direction_angle:.2f} degrees"
    )
    print(
        f"  thermal-energy flux     : "
        f"mean={flux_entity.mean():.6e}, "
        f"sum={flux_entity.sum():.6e}"
    )
    
#%%
# PLOT CONNECTED SHOCK ENTITIES

plt.figure(figsize=(7, 7))

for entity_id in range(1, num_components + 1):
    entity_mask = component_labels == entity_id

    plt.scatter(
        geometry_x_np[entity_mask],
        geometry_y_np[entity_mask],
        s=12,
        label=f"Entity {entity_id}",
    )

plt.title("Connected shock entities in X interaction")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()