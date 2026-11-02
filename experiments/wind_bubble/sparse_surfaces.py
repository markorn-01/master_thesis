"""CPU-only surface selection and sparse catalogs for the online bubble runner.

Rows follow np.argwhere(mask) order. Grid indices identify source cells within
one event, not persistent shock identities. No mesh or time tracking is added.
"""
from __future__ import annotations

import numpy as np


SCHEMA_VERSION = 2


def source_overlap_flags(upstream_radii, downstream_radii, positions_valid, injection_radius):
    """Classify known endpoints; false flags alone do not imply known geometry."""
    up, down = np.asarray(upstream_radii), np.asarray(downstream_radii)
    known = np.asarray(positions_valid, dtype=bool) & np.isfinite(up) & np.isfinite(down)
    return dict(sample_positions_valid=known,
                upstream_inside_injection=known & (up <= injection_radius),
                downstream_inside_injection=known & (down <= injection_radius))


def sampling_position_flags(positions, normals, post_distance, pre_distance,
                            grid_spacing, injection_radius):
    """Flag source overlap at actual profile endpoints; never filter samples.

    Distances are in cells from the refined surface position. Geometry can be
    known even if a thermodynamic sample is invalid. Missing distances produce
    unknown geometry, not evidence that endpoints are outside the source.
    """
    positions, normals = np.asarray(positions), np.asarray(normals)
    count = len(positions)
    upstream = np.full((count, 3), np.nan)
    downstream = np.full((count, 3), np.nan)
    known = np.zeros(count, dtype=bool)
    if (post_distance is None) != (pre_distance is None):
        raise ValueError('Supply both sampling distances or neither.')
    if post_distance is not None:
        post, pre = np.asarray(post_distance), np.asarray(pre_distance)
        if post.shape != (count,) or pre.shape != (count,):
            raise ValueError('Sampling distances must match the selected positions.')
        known = (np.isfinite(positions).all(axis=1) & np.isfinite(normals).all(axis=1)
                 & (np.linalg.norm(normals, axis=1) > 0)
                 & np.isfinite(post) & np.isfinite(pre) & (post >= 0) & (pre >= 0))
        upstream[known] = positions[known] + grid_spacing*pre[known, None]*normals[known]
        downstream[known] = positions[known] - grid_spacing*post[known, None]*normals[known]
    up_radius = np.linalg.norm(upstream - .5, axis=1)
    down_radius = np.linalg.norm(downstream - .5, axis=1)
    return dict(upstream_positions=upstream, downstream_positions=downstream,
                upstream_radii=up_radius, downstream_radii=down_radius,
                **source_overlap_flags(up_radius, down_radius, known, injection_radius))


def surface_sample_arrays(mask, offsets, direction, mach, flux, rho2, rho1,
                          sampling_valid, centers, grid_spacing, injection_radius, *,
                          post_distance=None, pre_distance=None):
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
    geometry = sampling_position_flags(
        positions, normals,
        None if post_distance is None else np.asarray(post_distance)[mask],
        None if pre_distance is None else np.asarray(pre_distance)[mask],
        grid_spacing, injection_radius)
    return dict(grid_indices=indices, positions=positions, normals=normals,
                radii=radii, mach=np.asarray(mach)[mask],
                thermalization_flux=np.asarray(flux)[mask],
                rho_upstream=upstream, rho_downstream=downstream,
                density_jump=ratio, sampling_valid=state_valid,
                density_jump_valid=density_valid,
                outside_injection=np.isfinite(radii) & (radii > injection_radius),
                **geometry)


def sparse_surface_catalog(time, mask, offsets, direction, mach, flux, rho2, rho1,
                           sampling_valid, centers, grid_spacing, injection_radius, *,
                           post_distance=None, pre_distance=None):
    """Return only detected samples outside injection, including invalid properties.

    A finite positive density ratio below one remains valid by the same numerical
    convention as the histograms. Flux validity is numerical, not a coverage or
    physical accuracy guarantee. Area weights and radial shock labels are absent.
    """
    samples = surface_sample_arrays(mask, offsets, direction, mach, flux, rho2, rho1,
                                    sampling_valid, centers, grid_spacing, injection_radius,
                                    post_distance=post_distance, pre_distance=pre_distance)
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
