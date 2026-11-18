"""Render the checked full sequence with explicit raw-point fallbacks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.wind_bubble.check_adaptive_mesh_times import verify_report


def load_sequence(directory, root):
    hashes = {}; report = verify_report(directory/'verification.json', hashes)
    metadata = json.loads((directory/'metadata.json').read_text())
    if metadata['status'] != 'complete': raise ValueError('Sequence construction is incomplete.')
    frames = []
    for event in report['events']:
        path = directory/f'mesh_{event:04d}.npz'; check_path = directory/f'event_{event:04d}.json'
        source = root/'online'/f'surfaces_{event:04d}.npz'
        for item in [path, check_path, source]:
            actual = hashlib.sha256(item.read_bytes()).hexdigest()
            if str(item) in hashes and hashes[str(item)] != actual: raise ValueError(f'Changed input: {item}')
            hashes[str(item)] = actual
        check = json.loads(check_path.read_text())
        with np.load(path) as data: mesh = {key: data[key].copy() for key in data.files}
        with np.load(source) as data:
            for key in ['positions', 'normals', 'mach', 'upstream_inside_any_source', 'downstream_inside_any_source']:
                np.testing.assert_array_equal(mesh['original_'+key], data[key])
            mesh['sources'] = data['source_positions'].copy()
            np.testing.assert_array_equal(mesh['time'], data['time'])
        if not np.isclose(check['summary']['time'], float(mesh['time']), atol=1e-7, rtol=0):
            raise ValueError('Event report and mesh time differ.')
        frames.append(dict(event=event, mesh=mesh, summary=check['summary'], groups=check['details']['groups']))
    return frames, hashes, report


def verify_encoded_frames(path, expected_headers, crop_height=90):
    """Check every decoded GIF's header against all expected frame headers."""
    from PIL import Image
    matches, durations, errors = [], [], []
    with Image.open(path) as image:
        if image.n_frames != len(expected_headers): raise ValueError('GIF frame count differs from saved events.')
        size = list(image.size)
        for index in range(image.n_frames):
            image.seek(index); image.load()
            header = np.asarray(image.convert('RGB'))[:crop_height].astype(np.int16)
            distances = [float(np.abs(header-e.astype(np.int16)).mean()) for e in expected_headers]
            match = int(np.argmin(distances))
            if match != index: raise ValueError(f'Encoded frame {index} matches header {match}.')
            matches.append(match); errors.append(distances[index]); durations.append(image.info.get('duration', 0))
    return dict(decoded_frames=len(matches), decoded_header_matches=matches, frame_size=size,
                frame_durations_ms=durations, playback_seconds=sum(durations)/1000,
                header_mean_palette_errors=errors)


def render(frames, output, fps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    palette = list(plt.get_cmap('tab20').colors)
    all_points = np.concatenate([frame['mesh']['original_positions'] for frame in frames]+[frames[0]['mesh']['sources']])
    low, high = all_points.min(axis=0)-.04, all_points.max(axis=0)+.04
    all_segments = np.concatenate([f['mesh']['original_segments'] for f in frames])
    if len(all_segments) and all_segments.max() >= len(palette):
        raise ValueError('Sequence exceeds the distinct segment-color palette.')
    fig = plt.figure(figsize=(13, 7.5), dpi=95)
    axes = [fig.add_subplot(1, 2, i+1, projection='3d') for i in range(2)]
    title = fig.suptitle('', fontsize=17, y=.97)
    fig.text(.5, .90, 'Original samples and checked smooth display | same z <= 0.5 cutaway, camera and bounds',
             ha='center', fontsize=11)
    note = fig.text(.035, .16, '', va='top', fontsize=10)
    fig.text(.035, .035, 'Colors identify geometric continuation segments; combinations receive new labels/colors. Black stars: wind sources.\n'
             'Gray points are outside approved mesh groups. Every frame is fitted independently; no temporal interpolation or smoothing.\n'
             'Mesh pieces are display geometry, not physical shock counts. Physical measurements and tracking decisions are unchanged.', fontsize=9)
    fig.subplots_adjust(left=.01, right=.985, top=.84, bottom=.22, wspace=.03)
    headers = [None]*len(frames)

    def update(index):
        frame = frames[index]; mesh = frame['mesh']; summary = frame['summary']
        p = mesh['original_positions']; labels = mesh['original_labels']; segments = mesh['original_segments']
        cut = p[:, 2] <= .5
        for column, ax in enumerate(axes):
            ax.clear()
            if column == 0:
                for segment in np.unique(segments):
                    selected = cut & (segments == segment)
                    color = palette[int(segment)] if segment >= 0 else '#858585'
                    ax.scatter(*p[selected].T, color=[color], s=3, depthshade=False, linewidths=0)
            else:
                v, f, groups = mesh['vertices'], mesh['faces'], mesh['vertex_groups']
                shown = f[(v[f, 2] <= .5).all(axis=1)]
                mapping = {row['group']: row['segment'] for row in frame['groups']}
                if len(shown):
                    colors = np.array([palette[mapping[int(group)]] for group in groups[shown[:, 0]]])
                    ax.add_collection3d(Poly3DCollection(v[shown], facecolors=colors, edgecolors=colors, linewidths=0,
                        shade=True, lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45)))
                fallback = cut & ~np.isin(labels, mesh['approved_groups'])
                ax.scatter(*p[fallback].T, color='#858585', s=4, depthshade=False)
            ax.scatter(*mesh['sources'].T, color='black', marker='*', s=45, depthshade=False)
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
                   xlabel='x', ylabel='y', zlabel='z', box_aspect=high-low,
                   title='Original detected samples' if column == 0 else 'Adaptive surfaces + raw-point fallback')
            ax.view_init(elev=25, azim=-35)
        title.set_text(f'Three-source shock evolution | simulation t = {float(mesh["time"]):.2f} | event {frame["event"]:02d}')
        minimum = min((g['fraction_within_quarter_cell'] for g in frame['groups']), default=None)
        quality = 'No retained groups to evaluate' if minimum is None else f'Lowest group agreement within 0.25 cell: {minimum:.1%}'
        if summary['self_intersecting']:
            status_note = 'Candidate mesh has a detected self-intersection; this event is shown as original points.'
        elif summary['status'] == 'partial_accuracy_fallback':
            status_note = 'Groups below the fixed accuracy target remain original points.'
        elif summary['status'].endswith('fallback'):
            status_note = 'The mesh did not pass all display checks; original points are retained.'
        elif summary['status'] == 'no_detections':
            status_note = 'Initial state: no detected surface samples.'
        else:
            status_note = 'Retained groups pass the display checks; samples outside these groups remain points.'
        note.set_text(f'{summary["samples"]:,} detected samples | approved mesh groups: {summary["approved_groups"]}/{summary["retained_groups"]} '
                      f'| raw-point fallback: {summary["fallback_points"]:,}\n'
                      f'{quality}. Source-overlap flags: upstream {summary["upstream_source_overlaps"]:,}, '
                      f'downstream {summary["downstream_source_overlaps"]:,}.\n'+status_note)
        fig.canvas.draw()
        headers[index] = np.asarray(fig.canvas.buffer_rgba())[:90, :, :3].copy()

    movie = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=False)
    movie.save(output/'adaptive_surface_evolution.gif', writer=PillowWriter(fps=fps), dpi=95)
    verification = verify_encoded_frames(output/'adaptive_surface_evolution.gif', headers)
    selected = {0, len(frames)-1, *[i for i, f in enumerate(frames) if f['event'] in [6, 8, 13, 14]
                                 or f['summary']['status'].endswith('fallback')]}
    for index in sorted(selected):
        update(index); fig.savefig(output/f'event_{frames[index]["event"]:04d}.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    times = [float(frame['mesh']['time']) for frame in frames]
    whole = [frame['summary']['sample_fraction_within_quarter_cell']*100 if frame['groups'] else np.nan for frame in frames]
    minimum = [min((g['fraction_within_quarter_cell'] for g in frame['groups']), default=np.nan)*100 for frame in frames]
    axes[0].plot(times, whole, 'o-', label='All retained samples')
    axes[0].plot(times, minimum, 's-', label='Lowest group agreement')
    axes[0].axhline(95, color='#b75b37', ls='--', label='Fixed per-group target')
    axes[0].set(ylabel='Samples within 0.25 cell (%)', ylim=(0, 102)); axes[0].legend(fontsize=9); axes[0].grid(alpha=.2)
    axes[1].plot(times, [f['summary']['retained_groups'] for f in frames], 'o-', label='Retained geometric groups')
    axes[1].plot(times, [f['summary']['approved_groups'] for f in frames], 's--', label='Approved mesh groups')
    axes[1].set(xlabel='Simulation time', ylabel='Groups'); axes[1].legend(fontsize=9); axes[1].grid(alpha=.2)
    fig.suptitle('Display checks over the saved sequence')
    fig.text(.07, .025, 'Agreement is measured against detected samples, not physical truth. Empty/no-group events have no agreement statistic.\n'
             'Group counts describe geometric detections. A successful frame does not validate topology or temporal flicker.', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .95)); fig.savefig(output/'display_checks_over_time.png', dpi=140); plt.close(fig)
    return verification, dict(bounds_low=low.tolist(), bounds_high=high.tolist(), camera=[25, -35], cutaway_z=.5)


def run(directory, root, output, fps=1):
    if not np.isfinite(fps) or fps <= 0: raise ValueError('FPS must be positive and finite.')
    if output.exists(): raise ValueError('Use a fresh animation directory.')
    frames, hashes, build_report = load_sequence(directory, root)
    output.mkdir(parents=True)
    encoded, view = render(frames, output, fps)
    from PIL import Image
    for path in output.glob('*.png'):
        with Image.open(path) as image: image.load()
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected: raise ValueError('Changed validated input.')
    sources = [Path(__file__), Path('experiments/wind_bubble/check_adaptive_mesh_times.py')]
    report = dict(passed=True, events=build_report['events'], full_saved_sequence=build_report['full_saved_sequence'],
        fps=fps, **encoded, view=view, empty_events=build_report['no_detection_events'],
        events_requiring_group_fallback=build_report['events_requiring_group_fallback'],
        physical_arrays_exact=True, grouping_changed=False, tracking_changed=False,
        temporal_smoothing=False, temporal_flicker_validated=False, physical_topology_established=False,
        input_sha256=hashes, source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        output_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir() if path.is_file()})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['passed', 'events', 'decoded_frames', 'playback_seconds', 'events_requiring_group_fallback']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('outputs/three_bubbles_iwr_compgpu7_n064'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=1)
    args = parser.parse_args()
    run(args.directory, args.root, args.output_dir, args.fps)
