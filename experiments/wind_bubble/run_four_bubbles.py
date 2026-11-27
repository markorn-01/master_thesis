"""Four interacting winds using the existing online diagnostics and tracker.

This fixed pilot is not a developed turbulent-box experiment. It changes the
source configuration, not the detector, association criteria or integrator.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.wind_bubble.four_bubble_setup import SOURCES, source_budget, weighted_budget
from experiments.wind_bubble.multi_source_setup import INJECTION_RADIUS


REFERENCE = Path('outputs/two_bubbles_iwr_compgpu9_n064/metadata.json')


def preflight(args):
    if (args.num_analysis < 3 or not np.isfinite(args.t_end) or args.t_end <= 0):
        raise ValueError('Require at least three analysis events and finite positive t-end.')
    if args.verify_offline and args.num_cells > 64:
        raise ValueError('Stored-state verification is restricted to at most 64 cubed.')
    reference_path = args.match_source_budget
    reference = json.loads(reference_path.read_text()) if reference_path is not None else None
    budget = source_budget(args.num_cells, reference)
    weighted = weighted_budget(budget, args.source_strengths)
    positions = np.asarray(SOURCES)
    separations = np.linalg.norm(positions[:, None]-positions[None, :], axis=-1)
    return dict(pilot='fixed_four_source_tetrahedron', num_cells=args.num_cells, num_sources=4,
                source_positions=SOURCES, injection_radius=INJECTION_RADIUS,
                minimum_source_center_separation=float(separations[np.triu_indices(4, 1)].min()),
                minimum_injection_boundary_clearance=float(np.minimum(positions, 1-positions).min()-INJECTION_RADIUS),
                source_budget=budget, weighted_budget=weighted,
                num_analysis=args.num_analysis, t_end=args.t_end,
                requested_times=np.linspace(0, args.t_end, args.num_analysis).tolist(),
                track_shocks=args.track_shocks, verify_offline=args.verify_offline,
                reference_path=str(reference_path) if reference_path is not None else None,
                reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest() if reference_path is not None else None,
                scope='Source/injection preflight only; no claim about collision time, physical ancestry, '
                      'late surface-boundary clearance or turbulent-box performance.')


def run(args):
    plan = preflight(args)
    if args.preflight_only:
        print(json.dumps(plan, indent=2))
        return plan
    # Keep preflight NumPy-only; import the JAX runner only for actual evolution.
    from experiments.wind_bubble.run_two_bubbles import run as run_simulation
    return run_simulation(args, source_positions=SOURCES, budget_builder=source_budget,
                          strength_builder=weighted_budget,
                          extra_source_files=[Path(__file__),
                              Path('experiments/wind_bubble/four_bubble_setup.py'),
                              Path('experiments/wind_bubble/multi_source_setup.py')])


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-cells', type=int, default=64)
    parser.add_argument('--num-analysis', type=int, default=17)
    parser.add_argument('--t-end', type=float, default=.16)
    parser.add_argument('--source-strengths', type=float, nargs=4, default=[1., 1., 1., 1.])
    parser.add_argument('--match-source-budget', type=Path, default=REFERENCE)
    parser.add_argument('--track-shocks', action='store_true')
    parser.add_argument('--verify-offline', action='store_true')
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--preflight-only', action='store_true',
                        help='Print source geometry, mask counts and prescribed budgets without importing JAX or creating outputs.')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/four_bubbles_n064'))
    return parser


if __name__ == '__main__':
    run(argument_parser().parse_args())
