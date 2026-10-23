# Shared shock analyzer: first validation result

## What changed

The existing per-snapshot hydrodynamic analysis was extracted into a reusable
single-state function. The offline single-bubble runner now calls that function.
The shock-finder physics, solver, callback scheduling and tracking are unchanged.

## Equivalence evidence

The reference is the original `analyze_shocks_3d` implementation from Git commit
`c6cda81c189d7d8c2fb9e8349664741681272aae`. Each input state was analyzed by the
historical implementation, the shared function, and the updated offline wrapper.
All **24 output fields match exactly**, including array shape, dtype and NaN
locations, in all three cases:

| Case | Detected surface cells | Valid profile-aware Mach samples | Median profile-aware Mach |
| --- | ---: | ---: | ---: |
| Body-diagonal planar shock, imposed Mach 5, 24³ | 837 | 837 | 4.99947 |
| Uniform gas, 32³ | 0 | 0 | — |
| Wind bubble, 32³, t = 0.20 | 872 | 872 | 4.76825 |

The bubble uses the existing hydrodynamic EI setup, with injection radius
0.0625 (two cells at 32³). It was evolved once; the analyzers received the same
final state. The figure shows all detected surface samples and a cell-weighted
Mach histogram. It is a low-resolution demonstration, not new convergence
evidence or a separately validated reverse-shock measurement.

Environment: local CPU, Python 3.10.11, JAX 0.6.2, Equinox 0.12.2. Complete
machine-readable case results are in `equivalence.json`.

The explicit local regression suite passed **72 tests** in 72.509 seconds,
covering the shared analyzer, continuous sampling, planar benchmarks, 3D
surface detection, thermalization, radial tracking, and resolution comparisons.

## What this establishes

Extracting the shared analysis preserves the numerical outputs of the existing
offline method on a controlled oblique shock, an empty detection, and an evolved
wind bubble. This provides a tested foundation for the next online-analysis step.

## Remaining work

- Run the explicit hydrodynamic regression suite in the remote `.venv-iwr` environment.
- Correct and verify callback timing and prepare unpadded states.
- Split JAX device calculations from host-side output handling.
- Compare online and offline diagnostics at matching times.
- Measure runtime and memory overhead before adding sparse export.

Reproduce from the repository root in a compatible environment:

```bash
python -m experiments.shock.check_analyzer_equivalence
```

No remote experiment, production GPU validation, or online callback validation
is claimed by this result.
