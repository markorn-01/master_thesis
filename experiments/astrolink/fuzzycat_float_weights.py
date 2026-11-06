"""Version-scoped experimental FuzzyCat adapter; installed package untouched."""
from __future__ import annotations
import importlib.metadata
import numpy as np


def floating_sample_weights(ordering,fuzzy_clusters,sample_numbers,n_samples):
    """Construct original-index group assignments and floating reciprocals.

    Equivalent to the 0.3.2 helper's intended reciprocal calculation, with a
    floating result array. Missing realizations keep weight zero. No changes
    to similarity, aggregation, selected intervals or normalization.
    """
    ordering=np.asarray(ordering);sample_numbers=np.asarray(sample_numbers)
    which=np.full(len(ordering),-1,dtype=np.int32)
    weights=np.zeros((n_samples,len(fuzzy_clusters)),dtype=np.float64)
    for group,(start,end) in enumerate(fuzzy_clusters):
        members=ordering[int(start):int(end)]
        which[members]=group
        samples,counts=np.unique(sample_numbers[members],return_counts=True)
        weights[samples,group]=1./counts
    return which,weights


def experimental_float_weight_class():
    """Return a local subclass replacing only the reciprocal-weight helper."""
    from fuzzycat import FuzzyCat
    version=importlib.metadata.version('fuzzycategories')
    if version!='0.3.2':raise RuntimeError(f'This isolated adapter was scoped to FuzzyCat 0.3.2, found {version}.')

    class FloatWeightFuzzyCat(FuzzyCat):
        _setupHierarchyInformation_njit=staticmethod(floating_sample_weights)

    return FloatWeightFuzzyCat
