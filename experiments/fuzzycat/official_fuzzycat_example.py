from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as datasets

from fuzzycat import FuzzyCat, FuzzyData, FuzzyPlots


def main() -> None:
    """Run the official FuzzyCat toy uncertainty experiment."""

    # Reproducible toy dataset.
    np.random.seed(0)

    background = np.random.uniform(
        low=-2.0,
        high=2.0,
        size=(1000, 2),
    )

    moons, _ = datasets.make_moons(
        n_samples=2000,
        noise=0.1,
        random_state=0,
    )
    moons -= np.array([[0.5, 0.25]])

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

    P = np.vstack([
        background,
        moons,
        gauss_1,
        gauss_2,
    ])

    print(f"Point-cloud shape: {P.shape}")

    # Homogeneous isotropic positional uncertainty.
    sigma = 0.1
    covP = sigma**2

    # Number of independently perturbed representations.
    n_samples = 100
    n_points = P.shape[0]

    clusters_directory = Path("Clusters")

    if not clusters_directory.exists():
        print("Generating AstroLink clusterings...")
        FuzzyData.clusteringsFromRandomSamples(
            P,
            covP,
            nSamples=n_samples,
        )
    else:
        print("Clusters/ already exists; reusing generated clusterings.")

    print("Running FuzzyCat...")

    fc = FuzzyCat(
        n_samples,
        n_points,
    )
    fc.run()

    print(f"Number of fuzzy clusters: {len(fc.fuzzyClusters)}")
    print(f"Stabilities: {fc.stabilities}")
    print(f"Membership array shape: {fc.memberships.shape}")

    # Plot the FuzzyCat memberships.
    FuzzyPlots.plotFuzzyLabelsOnX(
        fc,
        P,
        membersOnly=True,
    )

    # Rename the automatically generated image.
    source = Path("FuzzyLabels.png")
    destination = Path("fuzzycat.png")

    if source.exists():
        if destination.exists():
            destination.unlink()

        source.rename(destination)
        print(f"Saved plot as {destination}")
    else:
        print("Expected FuzzyLabels.png was not created.")
    


if __name__ == "__main__":
    main()