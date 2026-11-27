"""Fixed tetrahedral four-source pilot with explicit per-source budget matching."""
from experiments.wind_bubble import multi_source_setup as multi


# Binary-exact coordinates give identical discrete masks under reflection.
# All six center separations are sqrt(2)/4; the layout is noncoplanar.
SOURCES = ((.375, .375, .375), (.625, .625, .375),
           (.625, .375, .625), (.375, .625, .625))


def injection_counts(num_cells):
    return multi.injection_counts(num_cells, SOURCES)


def source_budget(num_cells, reference=None, mass_rate=.01, velocity=10., actual_counts=None):
    return multi.source_budget(num_cells, reference, mass_rate, velocity,
                               sources=SOURCES, actual_counts=actual_counts)


def weighted_budget(base, strengths=None):
    return multi.weighted_budget(base, strengths)
