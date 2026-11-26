"""Package existing collision evidence for physical interpretation by Tobias.

No simulation, detector evaluation, new physical classification, or message
sending occurs. PDF creation/QA requires reportlab, pypdf, and pypdfium2 in
addition to the established scientific environment.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.wind_bubble.compare_front_motion_resolution import (
    ROOTS, load_checked_run, validate_matching_setup,
)
from experiments.wind_bubble.review_collision_front_motion import sha


PACKAGE = Path('outputs/tobias_collision_checkpoint_2026_11_25')
PDF = Path('output/pdf/collision_validation_checkpoint_2026_11_25.pdf')
CONTROL = Path('outputs/shock_contact_hll_hllc_n128_n256')
COMPARISON = Path('outputs/collision_front_resolution_comparison_n128_n256/comparison')


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def prepare_inputs():
    datasets = [load_checked_run(root) for root in ROOTS]
    validate_matching_setup(*(d['meta'] for d in datasets))
    fingerprints = {p: h for d in datasets for p, h in d['hashes'].items()}
    for root in (CONTROL, COMPARISON):
        p = root/'verification.json'; fingerprints[str(p)] = sha(p)
    report = json.loads((CONTROL/'verification.json').read_text())
    if not report['experiment_completed'] or not report['pure_contacts_rejected']:
        raise ValueError('Require the completed solver comparison and passing negatives.')
    for p, h in report['saved_case_sha256'].items():
        if sha(p) != h:
            raise ValueError(f'Changed numerical control: {p}')
        fingerprints[p] = h
    for name in ('snapshots.csv', 'motion_checks.csv'):
        p = CONTROL/name; fingerprints[str(p)] = sha(p)
    snapshots = read_csv(CONTROL/'snapshots.csv')
    motion = read_csv(CONTROL/'motion_checks.csv')
    chosen = next(r for r in snapshots if r['num_cells'] == '128' and r['scheme'] == 'HLL'
                  and r['contact_only'] == 'False' and abs(float(r['time'])-.08) < 1.e-6)
    jump = next(r for r in motion if r['num_cells'] == '128' and r['scheme'] == 'HLL'
                and r['fit_events'] == '33' and abs(float(r['time'])-.08) < 1.e-6)
    if chosen['shock_located'] != 'True' or jump['compatible_10pct'] != 'False':
        raise ValueError('The selected known-shock example no longer supports the stated claim.')
    facts = dict(exact_control_mach=report['exact_mach'],
        returned_control_mach=float(chosen['measured_mach']),
        control_density_peak_error_percent=100*float(chosen['peak_density_relative_error']),
        independent_sampled_state_check_passed=False,
        pure_contact_checks=report['pure_contact_checks'],
        pure_contact_false_detections=0,
        solver_peak_errors_percent={n: {scheme: 100*float(next(r['peak_density_relative_error']
            for r in snapshots if r['num_cells'] == str(n) and r['scheme'] == scheme
            and r['contact_only'] == 'False' and abs(float(r['time'])-.08) < 1.e-6))
            for scheme in ('HLL', 'HLLC')} for n in (128, 256)},
        no_physical_absence_claim=True, no_256_production_mask_comparison=True)
    return datasets, report, facts, fingerprints


def plot_bubble_profiles(datasets, path):
    import matplotlib.pyplot as plt
    colors = {128: '#d47724', 256: '#087f8c'}
    times = (.04, .06, .08, .12)
    fig, axes = plt.subplots(3, 4, figsize=(14, 6.3), sharex=True, sharey='row')
    fig.subplots_adjust(left=.065, right=.99, bottom=.10, top=.90, wspace=.13, hspace=.11)
    for col, time in enumerate(times):
        for data in datasets:
            n = data['meta']['num_cells']; event = int(np.argmin(abs(data['times']-time)))
            if abs(data['times'][event]-time) > 1.e-6:
                raise ValueError('Requested figure time unavailable.')
            q = data['lines'][event]; x = data['x']
            visible = (x >= .43-1/n) & (x <= .57+1/n)
            for row, variable in enumerate((0, 4, 1)):
                axes[row, col].plot(x[visible], q[variable, visible], '.-', ms=3.2, lw=1.7,
                    color=colors[n], label=rf'${n}^3$')
        axes[0, col].set_title(f't = {time:.2f}', fontsize=12, pad=6)
        for row, ax in enumerate(axes[:, col]):
            ax.set_xlim(.43, .57); ax.set_xticks([.44, .48, .52, .56])
            ax.axvline(.5, color='.5', ls=':', lw=.9)
            ax.axvspan(.43, .4375, color='.85', alpha=.4, zorder=-2)
            ax.axvspan(.5625, .57, color='.85', alpha=.4, zorder=-2)
            ax.grid(axis='y', alpha=.16); ax.tick_params(labelsize=10)
            if row < 2:
                ax.set_yscale('log')
        axes[-1, col].set_xlabel('x (domain units)', fontsize=11)
    for ax, label in zip(axes[:, 0], ('Density', 'Pressure', 'x velocity')):
        ax.set_ylabel(label+' (code units)', fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12, frameon=False)
    fig.savefig(path, dpi=190, facecolor='white'); plt.close(fig)


def plot_known_control(path):
    import matplotlib.pyplot as plt
    from experiments.shock.shock_contact_solver_reference import ShockContactReference
    reference = ShockContactReference()
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 4.1))
    fig.subplots_adjust(left=.07, right=.99, bottom=.17, top=.85, wspace=.35)
    for scheme, color in (('HLL', '#d47724'), ('HLLC', '#087f8c')):
        with np.load(CONTROL/f'shock_contact_{scheme}_n0128.npz', allow_pickle=False) as data:
            event = int(np.argmin(abs(data['times']-.08)))
            q, exact, time = data['profiles'][event], data['exact_profiles'][event], float(data['times'][event])
        n = len(q); x = (np.arange(n)+.5)/n
        contact, shock = reference.positions(time)
        visible = (x >= contact-.028) & (x <= shock+.035)
        for ax, variable, label in zip(axes, (0, 2, 1), ('Density', 'Pressure', 'x velocity')):
            ax.plot(x[visible], q[visible, variable], '.-', ms=4, lw=1.7, color=color, label=scheme)
            if scheme == 'HLL':
                ax.plot(x[visible], exact[visible, variable], 'k--', lw=1.2, label='Exact cell averages')
            ax.axvline(contact, color='.65', ls=':', lw=.8)
            ax.axvline(shock, color='.3', ls=':', lw=.8)
            ax.set(xlim=(contact-.023, shock+.031), xlabel='x', title=label)
            ax.tick_params(labelsize=10); ax.grid(axis='y', alpha=.15)
    axes[0].set_ylim(0, 4.2)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [0, 2, 1]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
        loc='upper center', ncol=3, frameon=False, fontsize=10)
    fig.savefig(path, dpi=190, facecolor='white'); plt.close(fig)


def plot_solver_bars(facts, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    fig.subplots_adjust(left=.18, right=.98, bottom=.16, top=.85)
    for offset, scheme, color in ((-.19, 'HLL', '#d47724'), (.19, 'HLLC', '#087f8c')):
        values = [facts['solver_peak_errors_percent'][n][scheme] for n in (128, 256)]
        bars = ax.bar(np.arange(2)+offset, values, width=.34, label=scheme, color=color)
        ax.bar_label(bars, labels=[f'{v:.1f}' for v in values], padding=4, fontsize=11)
    ax.set(xticks=[0, 1], xticklabels=['128 cells', '256 cells'], ylim=(0, 35),
           ylabel='Density-peak error (%)')
    ax.tick_params(labelsize=11); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.22), ncol=2, frameon=False, fontsize=11)
    fig.savefig(path, dpi=190, facecolor='white'); plt.close(fig)


def create_pdf(pdf_path, images, facts):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph
    from PIL import Image

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = landscape(A4); margin = 34; available = width-2*margin
    c = canvas.Canvas(str(pdf_path), pagesize=(width, height), pageCompression=1)
    c.setTitle('Collision-front validation: evidence checkpoint, 25 November 2026')
    c.setAuthor('Minh Ngo')
    blue, ink, muted = '#087f8c', '#173240', '#536670'
    styles = {name: ParagraphStyle(name, fontName='Helvetica', fontSize=size,
        leading=leading, textColor=colors.HexColor(ink), alignment=TA_LEFT)
        for name, size, leading in (('body', 10, 12.5), ('caption', 9, 11.5), ('small', 8, 10))}

    def paragraph(text, x, top, box_width, style='body'):
        p = Paragraph(text, styles[style]); _, h = p.wrap(box_width, 1000)
        if top-h < 32:
            raise ValueError('PDF paragraph exceeds the safe page area.')
        p.drawOn(c, x, top-h)
        return top-h

    def header(kicker, title, subtitle, page):
        c.setFillColor(colors.HexColor(blue)); c.rect(margin, height-26, available, 3, fill=1, stroke=0)
        c.setFont('Helvetica-Bold', 8.5); c.drawString(margin, height-42, kicker)
        c.setFillColor(colors.HexColor(ink)); c.setFont('Helvetica-Bold', 18)
        c.drawString(margin, height-67, title)
        c.setFillColor(colors.HexColor(muted)); c.setFont('Helvetica', 9)
        c.drawString(margin, height-83, subtitle)
        c.setFont('Helvetica', 8); c.drawString(margin, 19, 'Thesis research checkpoint | 25 November 2026')
        c.drawRightString(width-margin, 19, f'{page} / 2')

    def picture(path, x, top, box_width, box_height):
        with Image.open(path) as im:
            iw, ih = im.size
        scale = min(box_width/iw, box_height/ih)
        w, h = iw*scale, ih*scale
        c.drawImage(str(path), x+(box_width-w)/2, top-h, width=w, height=h, mask='auto')

    def callout(text, top, box_height):
        c.setFillColor(colors.HexColor('#edf5f6'))
        c.roundRect(margin, top-box_height, available, box_height, 5, fill=1, stroke=0)
        bottom = paragraph(text, margin+10, top-8, available-20)
        if bottom < top-box_height+6:
            raise ValueError('Callout text does not fit.')

    header('ACTUAL FLOW / PHYSICAL QUESTION', 'What does the collision layer contain?',
        'Three-source 3D runs | matched physical setup and output schedule | same ray y=0.375, z=0.5', 1)
    callout('<b>Current limit:</b> The profiles sharpen with resolution, but we have not independently established '
            'which features in this cut are distinct reflected shock fronts.', height-97, 39)
    picture(images[0], margin, height-144, available, 349)
    paragraph('<b>Figure 1.</b> Orange: 128^3; teal: 256^3. Density and pressure use logarithmic axes; '
        'velocity is linear. The dotted line is the collision midplane, not a detected shock. '
        'Gray edge strips mark the prescribed injection regions on this ray.', margin, 92, available, 'caption')
    paragraph('The 256^3 run retained flow profiles only. It supplies no production detection-mask comparison. '
        'These curves alone do not establish shock absence or completeness.', margin, 58, available, 'small')
    c.showPage()

    header('KNOWN-ANSWER CONTROL / INTERPRETATION LIMIT', 'A failed sampled-state fit does not prove shock absence',
        'Exact shock plus nearby density contact | separate one-dimensional reference experiment', 2)
    c.setFillColor(colors.HexColor(ink)); c.setFont('Helvetica-Bold', 10)
    c.drawString(margin, height-107, 'Known Mach-8 shock: 128 cells, t=0.08')
    c.drawString(margin+500, height-107, 'HLLC gives a modest improvement')
    picture(images[1], margin, height-117, 480, 229)
    picture(images[2], margin+500, height-117, available-500, 229)
    paragraph('<b>Figure 2.</b> Dashed: exact conserved-cell averages. Dotted lines: exact contact and shock positions. '
        'The time shown belongs to this 1D control, not the bubble run.', margin, 240, 480, 'caption')
    paragraph('<b>Figure 3.</b> Peak-density errors at the same control time. '
        'The exact downstream density is 3.821.', margin+500, 240, available-500, 'caption')
    y = 190
    for text in (
        f'<b>Demonstrated:</b> The finder returns Mach {facts["returned_control_mach"]:.3f} for the exact Mach-8 shock, '
        f'although the recorded density peak is {facts["control_density_peak_error_percent"]:.1f}% low and the strict sampled-state check fails.',
        f'<b>Solver comparison:</b> HLLC reduces the selected peak errors only modestly. All '
        f'{facts["pure_contact_checks"]} pure-contact checks produce zero shock detections. This does not support another HLLC bubble trial.',
        '<b>What remains open:</b> This control explains a limitation of our validation procedure; '
        'it does not identify the physical waves in the three-dimensional collision layer.',
    ):
        y = paragraph(text, margin, y, available)-5
    callout('<b>Question:</b> In Figure 1, where do you expect the reflected shock fronts, '
        'and at which times should they remain distinct? Marking the fronts on the plots, '
        'or giving approximate positions and times, would help define a concrete validation target.', 102, 57)
    c.save()


def build(package=PACKAGE, pdf=PDF):
    package, pdf = Path(package), Path(pdf)
    if package.exists() or pdf.exists():
        raise ValueError('Use fresh package and PDF paths.')
    datasets, report, facts, hashes = prepare_inputs()
    package.mkdir(parents=True)
    images = [package/name for name in ('01_bubble_profiles.png', '02_known_shock_control.png', '03_solver_comparison.png')]
    plot_bubble_profiles(datasets, images[0]); plot_known_control(images[1]); plot_solver_bars(facts, images[2])
    create_pdf(pdf, images, facts)
    message = (
        'Hi Tobias, I followed up on the collision-front question with matched 128^3 and 256^3 flow profiles '
        'and an exact one-dimensional shock/contact control.\n\n'
        'Higher resolution sharpens the bubble profiles, but I have not yet established which features are distinct reflected shocks. '
        'The control exposed a limitation of the additional validation check: it can reject sampled states even when the finder '
        f'correctly detects a known Mach-8 shock and returns Mach {facts["returned_control_mach"]:.2f}. '
        'A failed sampled-state fit therefore cannot establish shock absence in a thin mixed layer. '
        'The HLL/HLLC comparison gave only modest improvement, so I am not proposing another solver-switch bubble run.\n\n'
        'I attached a short brief. Could you indicate where the reflected shock fronts should appear in Figure 1, '
        'and at which times they should remain distinct? A marked figure or a brief description of their positions '
        'and times would help define the physical validation target.\n'
    )
    (package/'message_to_tobias.txt').write_text(message)
    (package/'README.txt').write_text(
        'Evidence checkpoint for Tobias - 25 November 2026\n\n'
        f'Shareable two-page brief: {pdf}\n'
        'Optional standalone attachments: 01_bubble_profiles.png, 02_known_shock_control.png, 03_solver_comparison.png\n'
        'Copy-ready draft: message_to_tobias.txt\n\n'
        'Nothing was sent. No simulation or detector evaluation was performed.\n'
        'The actual-flow and exact-reference time axes belong to different experiments.\n'
        'The question concerns physical interpretation of the actual profiles, not approval of a detection claim.\n')
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise AssertionError(f'Input changed while packaging: {path}')
    manifest = dict(package_created=True, date='2026-11-25', simulation_rerun=False,
        message_sent=False, figures=len(images), facts=facts,
        input_sha256=hashes, build_source_sha256=sha(__file__),
        output_sha256={str(p): sha(p) for p in [pdf, *images, package/'message_to_tobias.txt']})
    (package/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(pdf=str(pdf), package=str(package), verified_inputs=len(hashes), facts=facts), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PACKAGE)
    parser.add_argument('--pdf-output', type=Path, default=PDF)
    args = parser.parse_args(); build(args.output_dir, args.pdf_output)
