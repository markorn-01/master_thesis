"""Audit the production finder on an independently known shock/contact fan.

No simulation is evolved and no production criterion is changed. Exact
conserved cell averages and explicitly synthetic broadening are kept separate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import jax
import numpy as np

from experiments.shock.check_colliding_shocks import fingerprint, make_analyzer, write_csv
from experiments.shock.shock_contact_reference import ShockContactReference

GATES = ('compression', 'gradient_alignment', 'pressure_jump_pass', 'temperature_jump_pass')


def pressure_transition(profile):
    """Independent footprint: cells whose centered pressure difference is nonzero.

    There is only one physical pressure jump in this reference. The contact
    has equal pressure on both sides. The tolerance excludes roundoff in the
    analytic conservative-to-primitive conversion; it is not a finder threshold.
    """
    pressure = np.asarray(profile)[:, 2]
    footprint = abs(np.gradient(pressure)) > 1e-10 * max(1., float(pressure.max()))
    footprint[:2] = False
    footprint[-2:] = False
    return footprint


def score(profile, fields, layout, resolution, kind):
    n = resolution
    center = n // 2
    line = {key: value[:, center, center] for key, value in fields.items() if value.ndim == 3}
    line['normal_x'] = fields['direction'][:, center, center, 0]
    x = (np.arange(n) + .5) / n
    band = pressure_transition(profile)
    row = dict(pressure_transition_cells=int(band.sum()))
    for key in (*GATES, 'shock_zone', 'shock_surface'):
        row[key + '_in_pressure_transition'] = int((line[key] & band).sum())
    patterns = sum(line[key].astype(np.uint8) * (1 << i) for i, key in enumerate(GATES))
    for code in range(16):
        row[f'pressure_cells_pass_pattern_{code:04b}'] = int((band & (patterns == code)).sum())
    assert sum(row[f'pressure_cells_pass_pattern_{c:04b}'] for c in range(16)) == int(band.sum())
    surface = fields['shock_surface'].copy()
    surface[:2] = surface[-2:] = False
    surface[:, :2] = surface[:, -2:] = False
    surface[:, :, :2] = surface[:, :, -2:] = False
    row['all_interior_surface_samples'] = int(surface.sum())
    row.update(coverage_1cell=None, coverage_2cell=None, mach_median=None,
               mach_relative_error=None, position_error_cells=None,
               surface_near_shock_without_identity_gate=None)
    if kind == 'contact':
        row['outcome'] = 'contact_rejected' if not surface.any() else 'contact_false_surface'
        return row, line, band

    normal = fields['direction'][..., 0]
    positions = x[:, None, None] + fields['surface_offsets'] * normal / n
    error = abs(positions - layout['shock_x']) * n
    oriented = normal * layout['orientation'] > .9
    # A contact close to a shock must not itself count as recovered shock.
    identity = np.ones(surface.shape, bool)
    if kind == 'pair':
        midpoint = .5 * (layout['shock_x'] + layout['contact_x'])
        identity = (positions - midpoint) * layout['orientation'] > 0
    row['surface_near_shock_without_identity_gate'] = int((surface & (error <= 2)).sum())
    for tolerance in (1, 2):
        selected = surface & identity & oriented & (error <= tolerance)
        row[f'coverage_{tolerance}cell'] = float(selected.any(axis=0)[2:-2, 2:-2].mean())
    selected = surface & identity & oriented & (error <= 2)
    values = fields['mach'][selected]
    values = values[np.isfinite(values) & (values > 0)]
    if len(values):
        row['mach_median'] = float(np.median(values))
        row['mach_relative_error'] = row['mach_median'] / layout['mach'] - 1
        row['position_error_cells'] = float(np.median(error[selected]))
    if not row['shock_zone_in_pressure_transition']:
        outcome = 'no_zone_candidates'
    elif not row['shock_surface_in_pressure_transition']:
        outcome = 'zone_but_no_surface'
    elif not row['coverage_2cell'] >= .99:
        outcome = 'surface_wrong_direction_or_position'
    elif row['mach_median'] is None:
        outcome = 'mach_unusable'
    elif abs(row['mach_relative_error']) > .05:
        outcome = 'mach_biased'
    else:
        outcome = 'recovered'
    row['outcome'] = outcome
    return row, line, band


def evaluate_case(analyze, reference, n, kind, width, phase, gap, output):
    if kind == 'pair':
        profile, layout = reference.profile(n, gap, width, phase)
    else:
        profile, layout = reference.control(n, kind, width, phase)
    factor = reference.contact_density_factor
    name = f'n{n:03d}_{kind}_c{factor:g}_s{gap:g}_f{width}_p{phase:g}'
    fields = analyze(profile)
    np.testing.assert_array_equal(fields['shock_zone'],
        np.logical_and.reduce([fields[key] for key in GATES]))
    summary, line, band = score(profile, fields, layout, n, kind)
    summary = dict(case=name, resolution=n, kind=kind, contact_density_factor=factor,
        separation_cells=gap if kind == 'pair' else None, filter_cells=width,
        shock_phase=phase, reference_mach=reference.mach if kind != 'contact' else None,
        shock_x=layout['shock_x'], contact_x=layout['contact_x'], **summary)
    np.savez_compressed(output/'profiles'/f'{name}.npz', **line,
        x=(np.arange(n)+.5)/n, density=profile[:, 0], velocity_x=profile[:, 1],
        pressure=profile[:, 2], temperature=profile[:, 2]/profile[:, 0],
        pressure_transition=band,
        shock_x=layout['shock_x'] if layout['shock_x'] is not None else np.nan,
        contact_x=layout['contact_x'] if layout['contact_x'] is not None else np.nan,
        reference_mach=layout['mach'] if kind != 'contact' else np.nan)
    return summary


def run(args):
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory.')
    if args.require_gpu and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU requested but unavailable.')
    resolutions = sorted(set(args.resolutions))
    widths = sorted(set(args.filter_cells))
    gaps = sorted(set(args.separations))
    factors = sorted(set(args.contact_factors))
    phases = sorted(set(args.phases))
    if not resolutions or any(n < 32 or n % 2 for n in resolutions):
        raise ValueError('Use even resolutions of at least 32.')
    if not widths or any(w < 1 or w % 2 != 1 for w in widths):
        raise ValueError('Filters must be positive odd integers.')
    if not gaps or any(not np.isfinite(g) or g <= 0 for g in gaps):
        raise ValueError('Separations must be finite and positive.')
    if not phases or any(not np.isfinite(p) or not 0 <= p < 1 for p in phases):
        raise ValueError('Phases must lie in [0,1).')
    if not factors or 1. in factors:
        raise ValueError('Use non-unit contact factors; isolated shocks are separate controls.')
    references = [ShockContactReference(args.mach, c) for c in factors]
    for ref in references:
        np.testing.assert_allclose(ref.flux_residuals(), 0, atol=1e-12, rtol=0)
    sources = [Path(__file__), Path('experiments/shock/shock_contact_reference.py'),
        Path('experiments/shock/colliding_shocks_reference.py'),
        Path('experiments/shock/check_colliding_shocks.py'),
        Path('experiments/wind_bubble/audit_detection_stages.py')]
    sources += sorted(Path('astronomix/_physics_modules/_shock_finder').glob('*.py'))
    hashes = {str(p): fingerprint(p) for p in sources}
    (args.output_dir/'profiles').mkdir(parents=True)
    rows = []
    for n in resolutions:
        analyze = make_analyzer(n)
        for width in widths:
            for phase in phases:
                rows.append(evaluate_case(analyze, references[0], n, 'shock', width, phase, 0., args.output_dir))
                for ref in references:
                    rows.append(evaluate_case(analyze, ref, n, 'contact', width, phase, 0., args.output_dir))
                    for gap in gaps:
                        rows.append(evaluate_case(analyze, ref, n, 'pair', width, phase, gap, args.output_dir))
            print(f'{n}³ filter={width}: contacts, isolated shock, and separation sweep complete.', flush=True)
    write_csv(args.output_dir/'summary.csv', rows)
    make_figures(args.output_dir, rows)
    for path, expected in hashes.items():
        if fingerprint(path) != expected:
            raise AssertionError(f'Source changed during benchmark: {path}')
    pair = [r for r in rows if r['kind'] == 'pair']
    exact_controls = [r for r in rows if r['kind'] == 'shock' and r['filter_cells'] == 1]
    report = dict(completed=True, backend=jax.default_backend(), jax=jax.__version__,
        production_changed=False, simulation_evolved=False,
        reference='Exact contact + right shock Riemann fan with zero-strength left acoustic wave.',
        gamma=references[0].gamma, mach=args.mach, cases=len(rows),
        pair_cases=len(pair), exact_pair_cases=sum(r['filter_cells'] == 1 for r in pair),
        outcome_counts=dict(Counter(r['outcome'] for r in pair)),
        exact_input_outcome_counts=dict(Counter(r['outcome'] for r in pair if r['filter_cells'] == 1)),
        isolated_exact_controls_pass=all(r['outcome'] == 'recovered' for r in exact_controls),
        contact_false_surface_samples=sum(r['all_interior_surface_samples'] for r in rows if r['kind'] == 'contact'),
        zone_rejected_pair_cases=[r['case'] for r in pair if r['outcome'] == 'no_zone_candidates'],
        contact_factors=factors, widths=widths, separations_cells=gaps, phases=phases,
        reference_states={str(c): dict(upstream=ref.upstream.tolist(), downstream=ref.downstream.tolist(),
            contact_gas=ref.contact_gas.tolist(), shock_speed=ref.shock_speed,
            contact_speed=ref.contact_speed, flux_residuals=ref.flux_residuals().tolist())
            for c, ref in zip(factors, references)}, source_sha256=hashes,
        scope='Known physical shock/contact fan, projected to a finite grid. Synthetic filters are not exact Euler solutions. '
              'A rejected close pair does not prove a resolved shock is missed in the bubble; no production criteria changed.',
        references=['https://www.clawpack.org/riemann_book/html/Euler.html'])
    (args.output_dir/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','reference_states','zone_rejected_pair_cases')}, indent=2))
    return report


def make_figures(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = max(r['resolution'] for r in rows)
    phase = min(r['shock_phase'] for r in rows)
    factors = sorted({r['contact_density_factor'] for r in rows if r['kind'] == 'pair'})
    strongest = max(factors)
    pairs = [r for r in rows if r['kind'] == 'pair' and r['resolution'] == n
             and r['contact_density_factor'] == strongest and r['shock_phase'] == phase]
    width = min(r['filter_cells'] for r in pairs)
    pairs = [r for r in pairs if r['filter_cells'] == width]
    chosen = [min(pairs, key=lambda r:r['separation_cells']), max(pairs, key=lambda r:r['separation_cells'])]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), dpi=150, sharex='col')
    fig.subplots_adjust(left=.09, right=.98, top=.88, bottom=.18, hspace=.32, wspace=.27)
    for col, row in enumerate(chosen):
        with np.load(output/'profiles'/f'{row["case"]}.npz') as d:
            x=d['x'];shock=float(d['shock_x']);contact=float(d['contact_x'])
            reference=ShockContactReference(row['reference_mach'],strongest)
            limits=(min(contact,shock)-3/n,max(contact,shock)+3/n)
            exact_x=np.array([limits[0],contact,shock,limits[1]])
            exact_states=np.vstack([reference.states,reference.upstream])
            exact_values=dict(pressure=exact_states[:,2],density=exact_states[:,0],
                              temperature=exact_states[:,2]/exact_states[:,0])
            for axis, key, label in zip(axes[:, col], ('pressure','density','temperature'), ('Pressure P','Density ρ','Temperature proxy P/ρ')):
                axis.step(exact_x,exact_values[key],where='post',ls='--',lw=1.2,color='#777777',label='Exact physical profile')
                axis.plot(x, d[key], 'o-', ms=3, lw=1, color='#087f8c',label='Analyzed grid values')
                axis.axvline(shock, ls='--', color='#bb4d34', label='Known shock')
                axis.axvline(contact, ls=':', color='#713e99', label='Known contact')
                axis.fill_between(x, 0, 1, where=d['shock_zone'], transform=axis.get_xaxis_transform(), color='#087f8c', alpha=.13)
                surface=d['shock_surface']
                axis.scatter(x[surface],d[key][surface],marker='v',s=45,color='#e68c18',zorder=5)
                axis.set(ylabel=label, xlim=limits)
                axis.grid(alpha=.2)
            description={'no_zone_candidates':'rejected at zone tests','recovered':'shock recovered'}.get(row['outcome'],row['outcome'])
            axes[0,col].set_title(f'Separation {row["separation_cells"]:g} cells: {description}')
            axes[2,col].set_xlabel('x')
            if col==0:
                trace=[]
                for i in np.flatnonzero(d['pressure_transition']):
                    trace.append(dict(case=row['case'],cell=int(i),**{key:d[key][i].item() for key in (
                        'x','pressure','density','temperature','velocity_x','divergence','normal_x',
                        'log_pressure_jump','log_temperature_jump',*GATES,'shock_zone','shock_surface')}))
                write_csv(output/'illustrative_zone_trace.csv',trace)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(f'Known Mach-{chosen[0]["reference_mach"]:g} shock beside a dense contact | {n}³, density factor {strongest:g}', fontsize=14)
    fig.text(.09,.05,'Gray dashed: exact Euler profile. Teal points: pressure/density/temperature derived from conservative grid averages (filter=1).\n'
        'Teal shading: detected zone. Orange markers: selected surface cells. The isolated shock and contacts are tested separately.\n'
        'Close-wave rejection on a coarse grid is not a proof of a resolved missed shock in the bubble.', fontsize=9)
    fig.savefig(output/'shock_contact_profiles.png');plt.close(fig)

    pair_rows=[r for r in rows if r['kind']=='pair' and r['resolution']==n and r['shock_phase']==phase]
    gaps=sorted({r['separation_cells'] for r in pair_rows})
    widths=sorted({r['filter_cells'] for r in pair_rows})
    codes={'recovered':0,'mach_biased':1,'mach_unusable':2,'surface_wrong_direction_or_position':3,
           'zone_but_no_surface':4,'no_zone_candidates':5}
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors=ListedColormap(['#147f72','#e7b44a','#bd8344','#8653a0','#b65a47','#4f5c6b'])
    fig,axes=plt.subplots(1,len(factors),figsize=(4.4*len(factors),3.8),dpi=150,squeeze=False)
    fig.subplots_adjust(left=.07,right=.99,top=.78,bottom=.30,wspace=.26)
    for ax,factor in zip(axes[0],factors):
        matrix=np.array([[codes[next(r['outcome'] for r in pair_rows if r['contact_density_factor']==factor
                                    and r['filter_cells']==w and r['separation_cells']==g)] for g in gaps] for w in widths])
        ax.imshow(matrix,aspect='auto',cmap=colors,norm=BoundaryNorm(np.arange(-.5,6.5),6),interpolation='nearest')
        ax.set(xticks=range(len(gaps)),xticklabels=[f'{g:g}' for g in gaps],
               yticks=range(len(widths)),yticklabels=widths,xlabel='Shock–contact separation (cells)',
               ylabel='Conservative filter width',title=f'Contact density factor {factor:g}')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=colors(i),label=name.replace('_',' ')) for name,i in codes.items()],
               loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,.01))
    fig.suptitle(f'Which stage limits recovery? | {n}³, shock phase {phase:g}',fontsize=14)
    fig.savefig(output/'recovery_by_separation.png');plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolutions', type=int, nargs='+', default=[32,64])
    parser.add_argument('--mach', type=float, default=3.)
    parser.add_argument('--contact-factors', type=float, nargs='+', default=[.25,4.,8.])
    parser.add_argument('--separations', type=float, nargs='+', default=[.5,1.,2.,4.,8.])
    parser.add_argument('--filter-cells', type=int, nargs='+', default=[1,3,5])
    parser.add_argument('--phases', type=float, nargs='+', default=[0.,.5])
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())
