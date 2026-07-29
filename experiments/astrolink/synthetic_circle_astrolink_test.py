import numpy as np
import matplotlib.pyplot as plt
from astrolink import AstroLink, visualize


def run_astrolink(P, title, adaptive=0, d_intrinsic=1):
    print(f"\n=== {title} ===")
    print("P shape:", P.shape)

    clusterer = AstroLink(
        P,
        adaptive=adaptive,
        d_intrinsic=d_intrinsic,
        verbose=1,
    )
    clusterer.run()

    print("Cluster IDs:", clusterer.ids)
    print("Number of hierarchy entries:", len(clusterer.clusters))
    print("Significances:", clusterer.significances)

    fig, ax = plt.subplots(figsize=(6, 6))
    visualize.labelsOnX(
        clusterer,
        P,
        skipZeroth=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.tight_layout()
    plt.show()

    return clusterer


# ---------------------------------------------------------------------------
# 1. Perfect uniformly sampled circle
# ---------------------------------------------------------------------------

n_points = 1200
radius = 0.35
center = np.array([0.5, 0.5])

theta = np.linspace(
    0.0,
    2.0 * np.pi,
    n_points,
    endpoint=False,
)

P_uniform_circle = np.column_stack([
    center[0] + radius * np.cos(theta),
    center[1] + radius * np.sin(theta),
])

# run_astrolink(
#     P_uniform_circle,
#     "Synthetic uniform circle",
# )


# # ---------------------------------------------------------------------------
# # 2. Noisy circle
# # ---------------------------------------------------------------------------

# rng = np.random.default_rng(0)

# noise_level = 0.005

# P_noisy_circle = P_uniform_circle + rng.normal(
#     scale=noise_level,
#     size=P_uniform_circle.shape,
# )

# run_astrolink(
#     P_noisy_circle,
#     "Synthetic noisy circle",
# )


# # ---------------------------------------------------------------------------
# # 3. Non-uniform circle: denser sampling on some arcs
# # ---------------------------------------------------------------------------

# theta_dense_1 = rng.uniform(0.0, 0.5 * np.pi, 400)
# theta_dense_2 = rng.uniform(np.pi, 1.5 * np.pi, 400)
# theta_sparse = rng.uniform(0.0, 2.0 * np.pi, 400)

# theta_nonuniform = np.concatenate([
#     theta_dense_1,
#     theta_dense_2,
#     theta_sparse,
# ])

# P_nonuniform_circle = np.column_stack([
#     center[0] + radius * np.cos(theta_nonuniform),
#     center[1] + radius * np.sin(theta_nonuniform),
# ])

# run_astrolink(
#     P_nonuniform_circle,
#     "Synthetic non-uniform circle",
# )


# # ---------------------------------------------------------------------------
# # 4. Cartesian-grid sampled ring, closer to shock-finder output
# # ---------------------------------------------------------------------------

# num_cells = 128
# x = (np.arange(num_cells) + 0.5) / num_cells
# y = (np.arange(num_cells) + 0.5) / num_cells
# X, Y = np.meshgrid(x, y, indexing="ij")

# R = np.sqrt(
#     (X - center[0]) ** 2
#     + (Y - center[1]) ** 2
# )

# ring_width = 1.0 / num_cells

# mask_ring = np.abs(R - radius) < ring_width

# P_grid_ring = np.column_stack([
#     X[mask_ring],
#     Y[mask_ring],
# ])

# run_astrolink(
#     P_grid_ring,
#     "Cartesian-grid sampled ring",
# )

# Same circle, but randomly shuffle point order
P_shuffled = P_uniform_circle.copy()
rng = np.random.default_rng(42)
rng.shuffle(P_shuffled)

run_astrolink(
    P_shuffled,
    "Synthetic uniform circle, shuffled order",
)