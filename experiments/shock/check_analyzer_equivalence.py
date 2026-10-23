"""Compare shared analysis with the pre-refactor implementation and plot a bubble.

Run: python -m experiments.shock.check_analyzer_equivalence
Requires local Git history containing the baseline commit. No remote access.
This checks extraction equivalence, not convergence or online callback behavior.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.shock import analyze_state as analyzer_module
from experiments.shock.analyze_state import analyze_shock_state
from experiments.shock.run_planar_shock_benchmark import build_planar_shock_state
from experiments.wind_bubble import run_single_bubble as bubble

BASELINE = "c6cda81c189d7d8c2fb9e8349664741681272aae"
ROOT = Path(__file__).resolve().parents[2]


def load_baseline():
    """Load only the original analyzer, without running the historical script."""
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:experiments/wind_bubble/run_single_bubble.py"],
        cwd=ROOT, text=True,
    )
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "analyze_shocks_3d")
    namespace = {**vars(analyzer_module), **vars(bubble)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "baseline_analyzer", "exec"), namespace)
    return namespace["analyze_shocks_3d"]


def compare_case(name, state, config, variables, helper, baseline):
    with contextlib.redirect_stdout(io.StringIO()):
        expected = baseline(np.asarray(state)[None], config, variables, helper)[0]
        wrapped = bubble.analyze_shocks_3d(np.asarray(state)[None], config, variables, helper)[0]
    actual = analyze_shock_state(state, config, variables, helper)
    assert actual.keys() == expected.keys() == wrapped.keys()
    for key in expected:
        # Require exact values, including matching NaN locations, shape and dtype.
        for candidate in (actual[key], wrapped[key]):
            assert candidate.dtype == expected[key].dtype, key
            np.testing.assert_array_equal(candidate, expected[key], err_msg=f"{name}: {key}")
    mask = actual["surface_mask"]
    mach = actual["profile_aware_mach_numbers"][mask]
    valid = np.isfinite(mach) & (mach > 0)
    if name == "uniform":
        assert not mask.any(), "Uniform gas must have no shock surface"
    else:
        assert mask.any() and valid.any(), f"{name}: empty comparison is not sufficient"
    report = dict(case=name, checked_fields=len(actual), exact_match=True,
                  surface_cells=int(mask.sum()), valid_profile_mach=int(valid.sum()),
                  profile_mach_median=float(np.median(mach[valid])) if valid.any() else None)
    print(report)
    return actual, report


def plot_example(result, output):
    mask = result["surface_mask"]
    points = result["refined_centers"][mask]
    mach = result["profile_aware_mach_numbers"][mask]
    valid = np.isfinite(mach) & (mach > 0)
    fig = plt.figure(figsize=(11, 4.8), constrained_layout=True)
    ax = fig.add_subplot(121, projection="3d")
    sc = ax.scatter(*points[valid].T, c=mach[valid], s=3, cmap="viridis")
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="Detected surface: profile-aware Mach")
    ax.set_box_aspect((1, 1, 1))
    fig.colorbar(sc, ax=ax, shrink=0.65, label="Mach number")
    ax = fig.add_subplot(122)
    ax.hist(mach[valid], bins=40, color="#287b8e")
    ax.set(xlabel="Profile-aware Mach number", ylabel="Surface-cell count",
           title="Full detected surface (cell-weighted)")
    fig.suptitle("Single wind bubble: 32³, t = 0.20 | extraction equivalence example")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shared_analyzer_validation"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline()
    reports = []
    # Original and shared code see the same controlled oblique shock state.
    state, config, variables, helper, _, _ = build_planar_shock_state(24, 5., 3., (1., 1., 1.))
    _, report = compare_case("oblique_planar", state, config, variables, helper, baseline)
    reports.append(report)
    setup = argparse.Namespace(num_cells=32, num_snapshots=3, t_end=0.2,
                               num_injection_cells=2, plasma_beta=None)
    state, config, params, variables, helper = bubble.build_problem(setup)
    _, report = compare_case("uniform", state, config, variables, helper, baseline)
    reports.append(report)
    # Only the final state is retained: no production GPU experiment is needed.
    config = config._replace(return_snapshots=False, print_elapsed_time=False, progress_bar=False)
    final_state = bubble.time_integration(state, config, params, variables)
    result, report = compare_case("wind_bubble_32", final_state, config, variables, helper, baseline)
    reports.append(report)
    plot_example(result, args.output_dir / "shock_surface_and_mach_histogram.png")
    import jax
    import equinox
    metadata = dict(baseline_commit=BASELINE, python=sys.version, jax=jax.__version__,
                    equinox=equinox.__version__, backend=jax.default_backend(),
                    comparison="Exact equality of every returned array, baseline vs shared and offline wrapper",
                    scope="Small-state extraction regression; not production convergence or online validation",
                    bubble=dict(num_cells=32, t_end=0.2, injection_radius=0.0625), cases=reports)
    (args.output_dir / "equivalence.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
