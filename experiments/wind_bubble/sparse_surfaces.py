"""CPU-only surface selection and sparse catalogs for the online bubble runner.

Rows follow np.argwhere(mask) order. Grid indices identify source cells within
one event, not persistent shock identities. No mesh or time tracking is added.
"""
from __future__ import annotations

import numpy as np


SCHEMA_VERSION = 1


def surface_sample_arrays(mask, offsets, direction, mach, flux, rho2, rho1,
                          sampling_valid, centers, grid_spacing, injection_radius):
    """Select detected cells once, using the established summary definitions."""
    mask = np.asarray(mask, dtype=bool)
    indices = np.argwhere(mask)
    normals = np.moveaxis(np.asarray(direction), 0, -1)[mask]
    positions = centers[mask] + grid_spacing*np.asarray(offsets)[mask, None]*normals
    radii = np.linalg.norm(positions - np.full(3, .5), axis=-1)
    downstream, upstream = np.asarray(rho2)[mask], np.asarray(rho1)[mask]
    state_valid = np.asarray(sampling_valid, dtype=bool)[mask]
    density_valid = (state_valid & np.isfinite(downstream) & np.isfinite(upstream)
                     & (downstream > 0) & (upstream > 0))
    ratio = np.full(downstream.shape, np.nan, dtype=float)
    np.divide(downstream, upstream, out=ratio, where=density_valid)
    return dict(grid_indices=indices, positions=positions, normals=normals,
                radii=radii, mach=np.asarray(mach)[mask],
                thermalization_flux=np.asarray(flux)[mask],
                rho_upstream=upstream, rho_downstream=downstream,
                density_jump=ratio, sampling_valid=state_valid,
                density_jump_valid=density_valid,
                outside_injection=np.isfinite(radii) & (radii > injection_radius))


def sparse_surface_catalog(time, mask, offsets, direction, mach, flux, rho2, rho1,
                           sampling_valid, centers, grid_spacing, injection_radius):
    """Return only detected samples outside injection, including invalid properties.

    A finite positive density ratio below one remains valid by the same numerical
    convention as the histograms. Flux validity is numerical, not a coverage or
    physical accuracy guarantee. Area weights and radial shock labels are absent.
    """
    samples = surface_sample_arrays(mask, offsets, direction, mach, flux, rho2, rho1,
                                    sampling_valid, centers, grid_spacing, injection_radius)
    selected = samples.pop('outside_injection')
    result = {key: value[selected] for key, value in samples.items()}
    result['grid_indices'] = result['grid_indices'].astype(np.int32)
    result['mach_valid'] = np.isfinite(result['mach']) & (result['mach'] > 0)
    result['thermalization_flux_valid'] = (np.isfinite(result['thermalization_flux'])
                                          & (result['thermalization_flux'] > 0))
    result.update(schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
                  time=np.asarray(time), grid_shape=np.asarray(np.shape(mask), dtype=np.int32),
                  grid_spacing=np.asarray(grid_spacing),
                  injection_radius=np.asarray(injection_radius),
                  domain_center=np.full(3, .5),
                  selection=np.asarray('detected surface cells with finite radius > injection_radius'),
                  units=np.asarray('dimensionless baseline code units'),
                  mach_estimator=np.asarray('profile-aware'))
    return result
