# Original versus corrected finder on the saved bubble collision

**Wider sampling does not fill the original central gap.** Ten saved GPU
states were compared on CPU, with no new hydrodynamic simulation.

| Time | Central slice, 64 cubed: baseline → new | Central slice, 128 cubed: baseline → new |
|---|---:|---:|
| 0.04 | 16 → 16 | 24 → 24 |
| 0.06 | 0 → 0 | 0 → 0 |
| 0.08 | 0 → 0 | 0 → 0 |
| 0.12 | 0 → 0 | 0 → 0 |
| 0.16 | 0 → 0 | 0 → 0 |

At 0.08 and 0.16, no cells in the central volume pass density/temperature
alignment. Some inspected late features also fail compression. Eight aligned,
compressive central-slice cells at 128 cubed and 0.12 still fail the wider jump
thresholds: pressure ratio at most 1.2214 versus 1.8625 required. Peripheral
changes do not establish recovered reflected collision fronts.

The frame-aware table classifies all ten states as **ambiguous** under the
available evidence. The strict outer-front controls fail, the 64-cubed captures
lack a matching high-cadence history, and the last saved time lacks a bracketed
speed fit. Supplemental 128-cubed strips also come from a separate-cadence
evolution. These limitations prohibit a claim of physical shock absence.

See the [method, explanation, and limits](../../experiments/wind_bubble/BUBBLE_JUMP_SCREENING_COMPARISON.md).

Figures:

- [128 cubed: density, surface changes, and early gating](detector_comparison_n128.png).
- [128 cubed: density, pressure, temperature, velocity, and stage flags](cross_layer_profiles_n128.png).
- [64 cubed: density and detector comparison](detector_comparison_n064.png).
- [64 cubed: native profiles and stage flags](cross_layer_profiles_n064.png).

Tables and reproducibility:

- `region_comparison.csv`: surface and stage counts in the central box and wider window.
- `native_axis_profiles.csv`: resolved native-axis gas and Boolean detector tests.
- `pressure_features.csv`: geometric inspection sites, not physical shock labels or temporal tracks.
- `frame_aware_classification.csv`: one row for each of the ten states, including controls, available speeds, Mach sensitivities, entropy, jump residuals, and classification.
- `frame_aware_feature_evidence.csv`: individual outer/central feature evidence.
- `extension_exception.json`: exact endpoint-test decomposition for the eight-cell exception.
- `historical_stage_comparison.csv`: historical GPU masks versus current default path.
- `halo_checks.csv`: exact Boolean and declared float-tolerance checks for 24/28-cell halos.
- `comparison_n*.npz`: cropped gas, coordinates, and detector-stage fields.
- `verification.json`, `frame_aware_verification.json`: scopes, limitations, and source/input fingerprints.

The detector comparison and 27 software tests complete successfully. The
frame-aware positive controls do **not** pass. The successful planar screening
correction is retained as opt-in; this bubble comparison is not a physical
completeness validation or a reason to relax the alignment criterion.
