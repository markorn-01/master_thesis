"""Small three-source experiment using the established online diagnostics."""
import argparse
from pathlib import Path
from experiments.wind_bubble.run_two_bubbles import run as run_simulation
from experiments.wind_bubble.three_bubble_setup import SOURCES,source_budget,weighted_budget


def run(args):
    return run_simulation(args,source_positions=SOURCES,budget_builder=source_budget,
        strength_builder=weighted_budget,extra_source_files=[Path(__file__),Path('experiments/wind_bubble/three_bubble_setup.py')])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-cells',type=int,default=64)
    parser.add_argument('--num-analysis',type=int,default=17)
    parser.add_argument('--t-end',type=float,default=.16)
    parser.add_argument('--source-strengths',type=float,nargs=3,default=[1.,1.,1.])
    parser.add_argument('--match-source-budget',type=Path,default=Path('outputs/two_bubbles_iwr_compgpu9_n064/metadata.json'))
    parser.add_argument('--verify-offline',action='store_true')
    parser.add_argument('--require-gpu',action='store_true')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run(args)
