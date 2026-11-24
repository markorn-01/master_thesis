# Transition-jump screening correction

See [implementation and validation scope](../../experiments/shock/TRANSITION_JUMP_SCREENING.md).

Replay of frozen, independently above-supported finite-blast events using the
baseline and opt-in transition-aware 3D hydrodynamic finder. Gas and independent
motion/jump labels are unchanged. These are repeated front observations, not
counts of distinct shocks.

| Cells | Supported front observations | Baseline misses | Corrected-mode misses |
|---|---:|---:|---:|
| 512 | 64 | 4 | 0 |
| 1024 | 98 | 12 | 0 |
| 2048 | 122 | 24 | 0 |

All 40 archived misses are recovered, with no previously detected supported
front lost. The default path reproduces the archived masks and floating-point
fields at the 142 replayed times exactly. The full archive has 903 times;
unsupported times were not silently treated as independent truth or rescored.

**Final verification passes every gate.** All 68 negative controls report zero
surfaces, no previously accurate opposing-front case regresses, and all 276
required separated-front checks on evolved gas pass.

The final reconstructed Mach cut is 1.3. Additional candidates and reported
surfaces must have increasing sampled log entropy above the fixed `1e-5`
roundoff floor. Surface entropy is evaluated at the same endpoints used for
Mach reconstruction. Existing default calls do not enable this mode.

Artifacts:

- `verification.json`: final pass/fail gates, fixed settings, source/input hashes.
- `front_comparison.csv`: all 284 independently above-supported front observations.
- `controls.csv`: 112 isolated shock/weak shock/contact/smooth/separated-pair controls.
- `opposing_controls.csv`: 84 opposing-front and isolated-branch control cases.
- `evolved_controls.csv`: 276 separated-front checks from genuinely evolved gas.
- `screening_correction.png`: recovered example and Mach comparisons without joining gaps.

No hydro evolution, new bubble run, runtime benchmark or change to the existing
online runner is part of this replay. Very close opposing fronts remain
unresolved in some controls. Passing these checks does not resolve the earlier
multi-bubble temperature-alignment rejection or establish universal completeness.

Local validation used the compatible Python 3.10/JAX 0.6.2 CPU environment.
The 125-test finder suite and updated 9-test focused suite passed. Remote GPU
validation and the saved bubble-state comparison remain separate follow-ups.
