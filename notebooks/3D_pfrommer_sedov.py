# ============================================================================
# 3D Shock Finder Test — Sedov-like point explosion
# ============================================================================
# This follows the structure of notebooks/2D_pfrommer_sedov.py, but the
# simulation, shock finder, radii, directions, and point cloud are all 3D.
#
# In the VS Code/Jupyter interactive window, run this in a separate cell before
# the plotting cells if you want a mouse-rotatable plot:
#
#     %matplotlib widget
#
# The notebook also writes a Plotly HTML file.  Unlike a PNG or the default
# inline Matplotlib rendering, that file remains mouse-rotatable in a browser.
#
# ============================================================================
# %%
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from astronomix import (
    CARTESIAN,
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import (
    find_shocks_pfrommer,
)
from astronomix.option_classes.simulation_config import HLLC, MINMOD

import astronomix
from astronomix._physics_modules._shock_finder._shock_surface import (
    _find_shock_surface_3d,
)

print("Using Astronomix from:", astronomix.__file__)
print("Loaded 3D surface function:", _find_shock_surface_3d)


# %%
# CONFIGURATION

num_cells = 64
box_size = 1.0
t_end = 0.05

config = SimulationConfig(
    geometry=CARTESIAN,
    dimensionality=3,
    riemann_solver=HLLC,
    limiter=MINMOD,
    box_size=box_size,
    num_cells=num_cells,
    mhd=False,
)
params = SimulationParams(gamma=5.0 / 3.0, t_end=t_end)

helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)

# geometric_centers has shape (nx, ny, nz, 3).
geometric_centers = helper_data.geometric_centers
geometry_x = geometric_centers[..., 0]
geometry_y = geometric_centers[..., 1]
geometry_z = geometric_centers[..., 2]


# ============================================================================
# INITIAL CONDITIONS — spherical point explosion
# ============================================================================

# %%
TARGET_CENTER = np.array([0.5, 0.5, 0.5])
center_x, center_y, center_z = TARGET_CENTER

E_explosion = 1.0
rho_ambient = 1.0
p_ambient = 1.0e-4
r_explosion = 0.05
gamma_gas = params.gamma

dx_from_center = geometry_x - center_x
dy_from_center = geometry_y - center_y
dz_from_center = geometry_z - center_z

# This is the real 3D radius.  The z term must not be omitted.
r = jnp.sqrt(
    dx_from_center**2
    + dy_from_center**2
    + dz_from_center**2
)
injection_mask = r < r_explosion

# Normalize by the volume actually occupied by grid cells.  This deposits
# exactly E_explosion as excess thermal energy on every resolution.
grid_spacing = box_size / num_cells
cell_volume = grid_spacing**3
injection_volume = jnp.sum(injection_mask) * cell_volume
pressure_excess = E_explosion * (gamma_gas - 1.0) / injection_volume

p = jnp.full_like(geometry_x, p_ambient)
p = p + jnp.where(injection_mask, pressure_excess, 0.0)
rho = jnp.full_like(geometry_x, rho_ambient)
u_x = jnp.zeros_like(geometry_x)
u_y = jnp.zeros_like(geometry_y)
u_z = jnp.zeros_like(geometry_z)

initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho,
    velocity_x=u_x,
    velocity_y=u_y,
    velocity_z=u_z,
    gas_pressure=p,
)
config = finalize_config(config, initial_state.shape)

deposited_energy = float(
    pressure_excess * injection_volume / (gamma_gas - 1.0)
)
print("=== 3D Sedov initial conditions ===")
print("State shape              :", initial_state.shape)
print("Grid spacing             :", float(config.grid_spacing))
print("Injection cells          :", int(jnp.sum(injection_mask)))
print("Discrete injection volume:", float(injection_volume))
print("Deposited thermal energy :", deposited_energy)


# ============================================================================
# RUN SIMULATION
# ============================================================================

# %%
final_state = time_integration(
    initial_state,
    config,
    params,
    registered_variables,
)

rho_final = final_state[registered_variables.density_index]
p_final = final_state[registered_variables.pressure_index]

if not bool(jnp.all(jnp.isfinite(final_state))):
    raise RuntimeError("The 3D Sedov simulation produced non-finite values.")

print("Final state shape:", final_state.shape)


# ============================================================================
# RUN THE FULL 3D SHOCK FINDER
# ============================================================================

# %%
result = find_shocks_pfrommer(
    final_state,
    config,
    registered_variables,
    helper_data,
)

surface_mask = np.asarray(result.shock_surface_cells, dtype=bool)
shock_zone_mask = np.asarray(result.shock_zones, dtype=bool)
mach = np.asarray(result.mach_numbers)
thermal_flux = np.asarray(result.thermal_energy_flux)

geometry_x_np = np.asarray(geometry_x)
geometry_y_np = np.asarray(geometry_y)
geometry_z_np = np.asarray(geometry_z)
r_np = np.asarray(r)
surface_offsets = np.asarray(result.shock_surface_offsets)
shock_direction = np.asarray(result.shock_direction)
grid_spacing = float(config.grid_spacing)

refined_x = geometry_x_np + grid_spacing * surface_offsets * shock_direction[0]
refined_y = geometry_y_np + grid_spacing * surface_offsets * shock_direction[1]
refined_z = geometry_z_np + grid_spacing * surface_offsets * shock_direction[2]
x_surface = refined_x[surface_mask]
y_surface = refined_y[surface_mask]
z_surface = refined_z[surface_mask]

# Every detected surface cell becomes one 3D point.
P = np.column_stack((x_surface, y_surface, z_surface))

print("\n=== 3D shock finder ===")
print("Shock-zone cells         :", int(shock_zone_mask.sum()))
print("Shock-surface cells      :", int(surface_mask.sum()))
print("Shock point-cloud shape  :", P.shape)
print(
    "Sub-cell offset range    :",
    (
        float(surface_offsets[surface_mask].min()),
        float(surface_offsets[surface_mask].max()),
    ),
)
if len(P) == 0:
    raise RuntimeError("No 3D shock-surface cells were detected.")


# ============================================================================
# FULL 3D GRAPH — mouse-rotatable with an interactive Matplotlib backend
# ============================================================================

# %%
mach_surface = mach[surface_mask]
mach_mean = float(np.mean(mach_surface))
mach_median = float(np.median(mach_surface))
mach_p16, mach_p84 = np.percentile(mach_surface, [16.0, 84.0])
mach_cv = float(np.std(mach_surface) / mach_mean)

print("\n=== Angular Mach-uniformity check ===")
print("Mean Mach              :", mach_mean)
print("Median Mach            :", mach_median)
print("16th–84th percentile   :", (mach_p16, mach_p84))
print("Coefficient of variation:", mach_cv)
print("Minimum–maximum Mach   :", (mach_surface.min(), mach_surface.max()))

figure = plt.figure(figsize=(9, 8))
axis = figure.add_subplot(111, projection="3d")
surface_plot = axis.scatter(
    x_surface,
    y_surface,
    z_surface,
    c=mach_surface,
    cmap="viridis",
    s=8,
    alpha=0.9,
)
axis.set(
    title="Complete 3D Sedov shock surface",
    xlabel="x",
    ylabel="y",
    zlabel="z",
    xlim=(0.0, box_size),
    ylim=(0.0, box_size),
    zlim=(0.0, box_size),
)
axis.set_box_aspect((1, 1, 1))
figure.colorbar(surface_plot, ax=axis, label="Mach number", shrink=0.7)
sedov_output_dir = Path("outputs/sedov_3d")
sedov_output_dir.mkdir(parents=True, exist_ok=True)
surface_figure_path = sedov_output_dir / "sedov_3d_mach_surface.png"
figure.savefig(surface_figure_path, dpi=180)
print("Saved static 3D Mach graph:", surface_figure_path.resolve())
plt.show()


# ============================================================================
# INTERACTIVE 3D GRAPH — open the HTML file in a browser and drag to rotate
# ============================================================================

# %%
interactive_figure = go.Figure(
    data=[
        go.Scatter3d(
            x=x_surface,
            y=y_surface,
            z=z_surface,
            mode="markers",
            marker={
                "size": 2.5,
                "color": mach_surface,
                "colorscale": "Viridis",
                "opacity": 0.85,
                "colorbar": {"title": "Mach number"},
            },
            customdata=np.linalg.norm(P - TARGET_CENTER, axis=1),
            hovertemplate=(
                "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}"
                "<br>Mach=%{marker.color:.3f}<br>radius=%{customdata:.4f}"
                "<extra></extra>"
            ),
        )
    ]
)
interactive_figure.update_layout(
    title="Interactive 3D Sedov shock surface",
    scene={
        "xaxis": {"title": "x", "range": [0.0, box_size]},
        "yaxis": {"title": "y", "range": [0.0, box_size]},
        "zaxis": {"title": "z", "range": [0.0, box_size]},
        "aspectmode": "cube",
    },
    width=900,
    height=800,
)

interactive_output = sedov_output_dir / "sedov_3d_interactive.html"
interactive_figure.write_html(interactive_output, include_plotlyjs=True)
print("Saved interactive 3D graph:", interactive_output.resolve())
try:
    get_ipython  # type: ignore[name-defined]
except NameError:
    # When run as a normal Python script, only save the HTML file.  This avoids
    # unexpectedly opening a browser during automated/headless executions.
    pass
else:
    interactive_figure.show()


# ============================================================================
# SHOCK-DIRECTION GRAPH
# ============================================================================

# %%
# shock_direction has shape (3, nx, ny, nz).  Move the component axis last.
shock_direction = np.moveaxis(np.asarray(result.shock_direction), 0, -1)
direction_surface = shock_direction[surface_mask]

# Compare the detected direction with the expected outward radial direction.
radial_vectors = P - TARGET_CENTER
radial_norms = np.linalg.norm(radial_vectors, axis=1, keepdims=True)
radial_unit_vectors = radial_vectors / np.maximum(radial_norms, 1.0e-30)
radial_alignment = np.sum(direction_surface * radial_unit_vectors, axis=1)

print("\n=== Shock-direction check ===")
print("Median radial alignment:", np.median(radial_alignment))
print("16th–84th percentile   :", np.percentile(radial_alignment, [16, 84]))
print("1 means perfectly outward; -1 means inward.")

# Plot only a subset of arrows so the graph remains readable.
arrow_step = max(1, len(P) // 250)
arrow_indices = np.arange(0, len(P), arrow_step)

figure = plt.figure(figsize=(9, 8))
axis = figure.add_subplot(111, projection="3d")
axis.scatter(P[:, 0], P[:, 1], P[:, 2], s=3, alpha=0.2)
axis.quiver(
    P[arrow_indices, 0],
    P[arrow_indices, 1],
    P[arrow_indices, 2],
    direction_surface[arrow_indices, 0],
    direction_surface[arrow_indices, 1],
    direction_surface[arrow_indices, 2],
    length=0.035,
    normalize=True,
    color="tab:red",
)
axis.set(
    title="3D shock-direction vectors",
    xlabel="x",
    ylabel="y",
    zlabel="z",
    xlim=(0.0, box_size),
    ylim=(0.0, box_size),
    zlim=(0.0, box_size),
)
axis.set_box_aspect((1, 1, 1))
plt.show()


# ============================================================================
# CENTRAL-SLICE CHECK — visualization only, not 2D shock finding
# ============================================================================

# %%
midplane = num_cells // 2
slab_start = max(0, midplane - 1)
slab_stop = min(num_cells, midplane + 2)
slab_surface = surface_mask[:, :, slab_start:slab_stop]
slab_x = geometry_x_np[:, :, slab_start:slab_stop][slab_surface]
slab_y = geometry_y_np[:, :, slab_start:slab_stop][slab_surface]

plt.figure(figsize=(7, 6))
plt.imshow(
    np.log10(np.maximum(np.asarray(p_final[:, :, midplane]), 1.0e-30)).T,
    origin="lower",
    extent=(0.0, box_size, 0.0, box_size),
    cmap="magma",
)
plt.scatter(
    slab_x,
    slab_y,
    s=14,
    facecolors="none",
    edgecolors="cyan",
    label="3D surface cells in central slab",
)
plt.colorbar(label=r"$\log_{10}(p)$")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Central pressure slice with nearby 3D detections")
plt.axis("equal")
plt.legend()
plt.show()


# ============================================================================
# RADIAL SHOCK MEASUREMENT AND SEDOV–TAYLOR COMPARISON
# ============================================================================

# %%
surface_radii = np.sqrt(
    (refined_x - TARGET_CENTER[0]) ** 2
    + (refined_y - TARGET_CENTER[1]) ** 2
    + (refined_z - TARGET_CENTER[2]) ** 2
)[surface_mask]
radius_p16, radius_median, radius_p84 = np.percentile(
    surface_radii, [16.0, 50.0, 84.0]
)

# For a 3D ideal-gas Sedov blast with gamma=5/3:
# R(t) = xi * (E * t^2 / rho_0)^(1/5), xi approximately 1.15167.
sedov_xi = 1.15167
analytic_radius = sedov_xi * (
    E_explosion * t_end**2 / rho_ambient
) ** 0.2
relative_radius_error = (radius_median - analytic_radius) / analytic_radius

print("\n=== Radial measurement ===")
print("16th-percentile radius  :", radius_p16)
print("Median detected radius  :", radius_median)
print("84th-percentile radius  :", radius_p84)
print("Analytic Sedov radius   :", analytic_radius)
print("Relative radius error   :", relative_radius_error)

plt.figure(figsize=(7, 5))
plt.hist(surface_radii, bins=35, alpha=0.8, label="3D surface cells")
plt.axvline(
    radius_median,
    color="tab:red",
    label="median detected radius",
)
plt.axvline(
    analytic_radius,
    color="black",
    linestyle="--",
    label="analytic Sedov radius",
)
plt.xlabel("3D radius from explosion centre")
plt.ylabel("shock-surface cell count")
plt.title("Full-volume radial distribution")
plt.legend()
plt.show()
