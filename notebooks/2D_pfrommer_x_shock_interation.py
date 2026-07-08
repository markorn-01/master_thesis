# ============================================================================
# 2D Shock Finder Test — Two Intersecting Shocks (X shape)
# ============================================================================
# Two Sod-like pressure discontinuities pass through (0.5, 0.5) at
# FRONT_NORMAL_ANGLE_1 = 60° and FRONT_NORMAL_ANGLE_2 = -20°, forming an X.
#
# Initial conditions:
#   Two signed distances (dist1, dist2) divide the domain into four wedges.
#   XOR: high pressure only where a cell is on opposite sides of the two fronts.
#   → two alternating high-pressure wedges, two low-pressure wedges.
#   High: p=1.0, ρ=1.0 — Low: p=0.1, ρ=0.125 — no initial velocity.
#
# Stress test goals (NOT a clean Sod validation):
#   1. Both shock arms detected away from the intersection
#   2. Shock finder does not crash or produce garbage everywhere
#   3. Detected d_s aligns with the expected normal along each arm
#   4. Intersection region (~r < 0.1) is ambiguous — noisy d_s expected there
# ==================================================================

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
geometry_x = geometric_centers[..., 0] # (nx, ny)
geometry_y = geometric_centers[..., 1] # (nx, ny)


# ============================================================================
# INITIAL CONDITIONS — double Sod (two outward-propagating shocks)
#
# Two planar fronts pass through center (0.5, 0.5) at +60° and -20°, forming an X shape
# Each front divides the domain into two half-planes via signed distance (dist1, dist2)
# XOR: a cell is high-pressure only if it's on opposite sides of the two fronts — this creates two alternating high/low pressure wedges
# High pressure: p=1.0, ρ=1.0 — Low pressure: p=0.1, ρ=0.125 (standard Sod values)
# No initial velocity anywhere
# The intersection region at the center is interensted
# ============================================================================

FRONT_NORMAL_ANGLE_1 =  60.0    # degrees — normal 
FRONT_NORMAL_ANGLE_2 = -20.0    # degrees — normal of shock 2
# both pass through the center
TARGET_CENTER = (0.5, 0.5)
target_theta1 = jnp.deg2rad(FRONT_NORMAL_ANGLE_1)
target_theta2 = jnp.deg2rad(FRONT_NORMAL_ANGLE_2)

target_nx_hat_1, target_ny_hat_1 = jnp.cos(target_theta1), jnp.sin(target_theta1)
target_nx_hat_2, target_ny_hat_2 = jnp.cos(target_theta2), jnp.sin(target_theta2)

# signed distance from each front
dist1 = (geometry_x - TARGET_CENTER[0]) * target_nx_hat_1 + (geometry_y - TARGET_CENTER[1]) * target_ny_hat_1   # signed dist from shock 1
dist2 = (geometry_x - TARGET_CENTER[0]) * target_nx_hat_2 + (geometry_y - TARGET_CENTER[1]) * target_ny_hat_2   # signed dist from shock 2

high1 = dist1 < 0
high2 = dist2 < 0

# XOR: high pressure in alternating wedges → pressure jump along both diagonals
in_high = jnp.logical_xor(high1, high2)

p   = jnp.where(in_high, 1.0, 0.1  )
rho = jnp.where(in_high, 1.0, 0.125)
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