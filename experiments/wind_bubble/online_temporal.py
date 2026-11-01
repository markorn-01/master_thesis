"""CPU-side single-bubble tracking and streaming energy diagnostics.

Reuse the offline radial statistics, continuity and confidence prescriptions.
No finder, integrator or per-cell identity matching is introduced here.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.wind_bubble.run_single_bubble import (
    split_radial_shock_candidates, _radial_band_statistics,
    _temporal_tracking_diagnostics, _shock_tracking_confidence,
)


class RadialShockTracker:
    """Maintain the offline forward/reverse candidate histories one event at a time."""
    def __init__(self, grid_spacing, injection_radius):
        self.grid_spacing = grid_spacing
        self.injection_radius = injection_radius
        self.event_index = 0
        self.last_time = None
        self.states = {kind: dict(last_radius=np.nan, last_radius_uncertainty=np.nan,
                                  last_time=np.nan, last_velocity=np.nan, last_snapshot=None)
                       for kind in ('reverse', 'forward')}

    def update(self, time, samples):
        time = float(time)
        if not np.isfinite(time) or (self.last_time is not None and time <= self.last_time):
            raise ValueError('Tracking times must be finite and strictly increasing.')
        outside = samples['outside_injection']
        radii, mach = samples['radii'][outside], samples['mach'][outside]
        radial = samples['positions'][outside] - .5
        unit_radial = radial / np.maximum(np.linalg.norm(radial, axis=-1)[:, None], 1e-30)
        alignment = np.sum(samples['normals'][outside] * unit_radial, axis=-1)
        reverse, forward, separation = split_radial_shock_candidates(
            radii, self.injection_radius, self.grid_spacing)
        statistics = {kind: _radial_band_statistics(band, radii, mach, alignment)
                      for kind, band in [('reverse', reverse), ('forward', forward)]}
        r, f = (float(statistics[k]['radius_median']) for k in ('reverse', 'forward'))
        ordering_ok = not (np.isfinite(r) and np.isfinite(f)) or r < f
        rows = []
        for kind, stats in statistics.items():
            state = self.states[kind]
            uncertainty = .5 * (float(stats['radius_p84']) - float(stats['radius_p16']))
            gap = self.event_index-state['last_snapshot'] if state['last_snapshot'] is not None else 1
            tracking = _temporal_tracking_diagnostics(
                float(stats['radius_median']), float(state['last_radius']), uncertainty,
                float(state['last_radius_uncertainty']), time, float(state['last_time']),
                float(state['last_velocity']), gap, self.grid_spacing)
            confidence = _shock_tracking_confidence(
                kind, stats, tracking, self.injection_radius, self.grid_spacing, ordering_ok)
            rows.append(dict(time=time, event_index=self.event_index,
                             track_id=f'{kind}_shock', shock_kind=kind,
                             two_bands_detected=bool(reverse.size), separation_radius=separation,
                             radial_ordering_ok=bool(ordering_ok), **stats, **tracking,
                             **{k: v for k, v in confidence.items() if k not in tracking}))
            # Preserve offline behavior: flag a discontinuity but update the
            # candidate history with every detected band. No automatic reassignment.
            if tracking['detected']:
                state.update(last_radius=float(stats['radius_median']),
                             last_radius_uncertainty=uncertainty, last_time=time,
                             last_velocity=float(tracking['radial_velocity']),
                             last_snapshot=self.event_index)
        self.last_time = time
        self.event_index += 1
        return rows


class CumulativeEnergy:
    """O(1)-state equivalent of _cumulative_trapezoid_over_detections.

    Missing intervals carry the accumulated value once initialized, but add no
    energy and no covered duration. This does not assert zero physical heating.
    """
    def __init__(self):
        self.previous_time = None
        self.previous_rate = np.nan
        self.total = np.nan
        self.first_detection_time = np.nan
        self.integrated_duration = 0.

    def update(self, time, rate):
        time, rate = float(time), float(rate)
        if not np.isfinite(time) or (self.previous_time is not None and time <= self.previous_time):
            raise ValueError('Energy times must be finite and strictly increasing.')
        valid_interval = False
        if np.isfinite(rate):
            if not np.isfinite(self.total):
                self.total = 0.
                self.first_detection_time = time
            elif np.isfinite(self.previous_rate):
                duration = time - self.previous_time
                self.total += .5 * (self.previous_rate+rate) * duration
                self.integrated_duration += duration
                valid_interval = True
        self.previous_time, self.previous_rate = time, rate
        return dict(dissipation_rate=rate, integration_interval_valid=valid_interval,
                    cumulative_dissipated_energy=float(self.total),
                    first_power_detection_time=float(self.first_detection_time),
                    integrated_duration=float(self.integrated_duration))


class TemporalDiagnostics:
    """Optional CSV products, with separate state for each execution workflow."""
    def __init__(self, directory, grid_spacing, injection_radius, *,
                 track_shocks=False, integrate_energy=False):
        self.directory = Path(directory)
        self.track_shocks = track_shocks
        self.integrate_energy = integrate_energy
        self.tracker = RadialShockTracker(grid_spacing, injection_radius) if track_shocks else None
        self.energy = {k: CumulativeEnergy() for k in ('reverse', 'forward')}
        self.track_rows, self.energy_rows = [], []
        self.streams, self.writers = {}, {}

    def _write(self, name, rows):
        for row in rows:
            if name not in self.writers:
                self.streams[name] = (self.directory/f'{name}.csv').open('x', newline='')
                self.writers[name] = csv.DictWriter(self.streams[name], fieldnames=list(row))
                self.writers[name].writeheader()
            self.writers[name].writerow(row)
        if rows:
            self.streams[name].flush()

    def receive(self, time, samples, summary):
        if self.track_shocks:
            rows = self.tracker.update(time, samples)
            self._write('tracks', rows)
            self.track_rows.extend(rows)
        if self.integrate_energy:
            row = dict(time=float(time), event_index=len(self.energy_rows))
            for kind, accumulator in self.energy.items():
                measurements = accumulator.update(time, summary[f'{kind}_dissipation_rate'])
                row.update({f'{kind}_{k}': v for k, v in measurements.items()})
            values = np.asarray([row[f'{k}_cumulative_dissipated_energy'] for k in self.energy])
            row['combined_cumulative_dissipated_energy'] = (
                float(np.nansum(values)) if np.isfinite(values).any() else np.nan)
            self._write('energy', [row])
            self.energy_rows.append(row)

    def close(self):
        for stream in self.streams.values():
            stream.close()


def compare_records(actual, expected, label):
    """Compare typed auxiliary rows, including exact identities/status/flags."""
    if len(actual) != len(expected):
        raise AssertionError(f'{label}: row counts differ.')
    for index, (a, b) in enumerate(zip(actual, expected)):
        if a.keys() != b.keys():
            raise AssertionError(f'{label}: schemas differ.')
        for key in a:
            if isinstance(b[key], (str, bool, int)):
                if a[key] != b[key]:
                    raise AssertionError(f'{label} row {index}, {key}: mismatch.')
            else:
                np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-8,
                                           equal_nan=True, err_msg=f'{label} row {index}, {key}')


def plot_temporal_histories(temporal, path):
    """Show candidate motion and accumulated finite-interval thermalization."""
    import matplotlib.pyplot as plt

    panels = int(temporal.track_shocks) + int(temporal.integrate_energy)
    if not panels:
        return
    fig, axes = plt.subplots(panels, 2, figsize=(11, 4*panels), squeeze=False,
                             constrained_layout=True)
    panel = 0
    if temporal.track_shocks:
        for kind in ('reverse', 'forward'):
            rows = [r for r in temporal.track_rows if r['shock_kind'] == kind]
            times = [r['time'] for r in rows]
            axes[panel, 0].plot(times, [r['radius_median'] for r in rows], 'o-', label=kind)
            axes[panel, 1].plot(times, [r['radial_velocity'] for r in rows], 'o-', label=kind)
        axes[panel, 0].set(ylabel='Candidate radius (code units)')
        axes[panel, 1].set(ylabel='Radial velocity (code units)')
        panel += 1
    if temporal.integrate_energy:
        rows = temporal.energy_rows
        times = [r['time'] for r in rows]
        for kind in ('reverse', 'forward'):
            rates = np.asarray([r[f'{kind}_dissipation_rate'] for r in rows])
            cumulative = np.asarray([r[f'{kind}_cumulative_dissipated_energy'] for r in rows])
            axes[panel, 0].plot(times, rates, 'o-', label=kind)
            axes[panel, 1].plot(times, np.where(np.isfinite(rates), cumulative, np.nan),
                               'o-', label=kind)
        axes[panel, 0].set(ylabel='Thermalization power (code units)')
        axes[panel, 1].set(ylabel='Cumulative dissipated energy (code units)')
    for axis in axes.flat:
        axis.set_xlabel('Simulation time')
        axis.grid(alpha=.25)
        axis.legend()
    fig.suptitle('Single-bubble temporal diagnostics')
    fig.supxlabel('Candidate labels use radial ordering. Energy integrates consecutive finite powers only.\n'
                  'Missing intervals are not bridged; continuity warnings are advisory.', fontsize=9)
    fig.savefig(path, dpi=170)
    plt.close(fig)
