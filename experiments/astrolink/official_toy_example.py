import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from astrolink import AstroLink, visualize


def main() -> None:
    """Run AstroLink on the official toy dataset and visualize the result."""

    # Make the random background reproducible.
    np.random.seed(0)

    # Uniform background noise.
    background = np.random.uniform(
        low=-2,
        high=2,
        size=(1000, 2),
    )

    # Two interleaving moon-shaped structures.
    moons, _ = datasets.make_moons(
        n_samples=2000,
        noise=0.1,
        random_state=0,
    )

    # Move the moons approximately to the centre.
    moons -= np.array([0.5, 0.25])

    # Two compact Gaussian clusters.
    gauss_1 = np.random.normal(
        loc=-1.25,
        scale=0.2,
        size=(500, 2),
    )

    gauss_2 = np.random.normal(
        loc=1.25,
        scale=0.2,
        size=(500, 2),
    )

    # Combine all point sets into one point-cloud array.
    P = np.vstack([
        background,
        moons,
        gauss_1,
        gauss_2,
    ])

    print(f"Input shape: {P.shape}")

    # Run AstroLink.
    clusterer = AstroLink(
        P,
        verbose=1,
    )
    clusterer.run()
    
    cluster_members = [
        clusterer.ordering[start:end]
        for start, end in clusterer.clusters
    ]

    for cluster_id, members, significance in zip(
        clusterer.ids,
        cluster_members,
        clusterer.significances,
    ):
        print(
            f"Cluster {cluster_id}: "
            f"{len(members)} points, "
            f"significance={significance:.3f}"
        )

    print(f"Number of returned clusters: {len(clusterer.clusters)}")
    print(f"Cluster IDs: {clusterer.ids}")
    print(f"Cluster significances: {clusterer.significances}")

    # Create one figure for the ordered-density plot.
    fig_density, ax_density = plt.subplots(figsize=(10, 5))

    visualize.orderedDensity(
        clusterer,
        skipZeroth=False,
        ax=ax_density,
    )

    ax_density.set_title(
        "AstroLink ordered-density plot"
    )
    ax_density.set_xlabel(
        "Position in AstroLink ordering"
    )
    ax_density.set_ylabel(
        "Normalized log-density"
    )

    fig_density.tight_layout()
    fig_density.savefig(
        "ordered_density.png",
        dpi=200,
        bbox_inches="tight",
    )

    # Create a separate figure for the point labels.
    fig_labels, ax_labels = plt.subplots(figsize=(8, 8))

    visualize.labelsOnX(
        clusterer,
        P,
        skipZeroth=False,
        ax=ax_labels,
        scatterKwargs={
            "s": 8,
            "edgecolor": "black",
            "linewidth": 0.1,
        },
    )

    ax_labels.set_title(
        "Official AstroLink toy example"
    )
    ax_labels.set_xlabel("x")
    ax_labels.set_ylabel("y")
    ax_labels.set_aspect("equal")
    ax_labels.legend(
        title="Cluster ID",
        framealpha=1,
    )

    fig_labels.tight_layout()
    fig_labels.savefig(
        "astrolink_labels.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

if __name__ == "__main__":
    main()