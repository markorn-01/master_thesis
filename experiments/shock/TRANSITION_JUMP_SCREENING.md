# Bounded jump-screening correction

## Purpose

The [finite-blast experiment](FINITE_BLAST_COLLISION.md) found reflected fronts
with independent motion/jump support above Mach 1.3 but no nearby shock-zone
candidates. The one-cell preliminary pressure test measured only part of the
broadened transition. This follow-up changes **how additional candidates are
screened**, while retaining the existing temperature direction, compression and
alignment requirements, surface selection, and pressure-to-Mach estimator.

This is an **explicit opt-in hydrodynamic mode**. Existing simulations and the
shared online analyzer continue to use the established default. It is not a
claim that the original multi-bubble temperature-alignment rejection is fixed.

## Implementation in plain language

1. Find compressive cells with aligned density/temperature gradients, before
   demanding the minimum jump. This is a provisional transition mask.
2. From each provisional cell, follow the local normal in both directions to
   the first exit from that mask, up to eight cell widths per side. The existing
   sampler searches in quarter-cell steps, adds its quarter-cell exterior
   margin, respects domain boundaries, and stops before an opposing normal.
3. At these endpoints, require both the pressure and temperature jumps for
   Mach 1.3. Also require an increasing sampled entropy proxy. Pressure and
   temperature use the same sampled states; they cannot pass at different
   distances chosen separately for each test.
4. Add passing cells to the existing one-cell jump candidates. Continue with
   the existing maximum-compression surface selection and Mach reconstruction.
5. Before reporting a surface, require valid samples, reconstructed Mach at
   least 1.3, and increasing sampled entropy at the **same endpoints selected
   for that Mach reconstruction**. Rejected surfaces have zero Mach, flux,
   offsets and IDs, and are excluded from the reported sample count.

The zone mask remains a candidate region: a zone can survive while its final
surface fails the strength/entropy acceptance. Zones are not reported physical
shock identities.

The sampled entropy proxy is `log(P) - (5/3)*log(rho)`. Its required increase
is greater than `1e-5`, a fixed numerical roundoff guard. We interpolate this
cell entropy field at the chosen endpoints. Computing entropy after separately
interpolating pressure and density can introduce a spurious change in an
exactly isentropic profile. The entropy check is useful here but does not prove
that every accepted structure is a shock.

The first extension-only prototype recovered the archived misses but left a
baseline false detection in a steep isentropic ramp. It was not accepted.
The final surface entropy check rejects that control. The numerical entropy
floor and Mach threshold were not tuned to individual blast events.

## API and scope

```python
result = find_shocks_pfrommer(
    state, config, registered_variables, helper_data,
    mach_min=1.3,
    profile_aware_mach=True,
    transition_aware_detection=True,
)
```

`transition_aware_detection=False` remains the default. The demonstrated
correction uses the profile-aware estimator; callers choosing the first-outside
estimator still receive the final threshold/entropy checks but should not assume
the same completeness. The MHD finder is unchanged.

This screening extension follows the existing separation between shock zones,
surfaces, and reconstructed jumps. See [Schaal & Springel (2015)](https://doi.org/10.1093/mnras/stu2386)
for that framework; the bounded provisional-mask extension and entropy gates
here are our implementation choices, not a claim of reproducing that paper's
minimum-jump implementation exactly.

## Bounded validation

Run `check_transition_jump_screening.py` against the frozen finite-blast archive.
It does not evolve the gas again or change the independently formed labels.
It compares the default path with archived masks and floating-point fields at
every independently above-supported event, then compares the new mode on the
same profiles. Repeated front-events are measurements, not distinct shocks.

Additional controls include isolated Mach 1.2/1.5/3/10 shocks with several
synthetic widths and both orientations; moving contacts; steep and broad
isentropic compression; separated shock/contact pairs; incoming/outgoing
opposing fronts across a separation/width sweep; and the previous genuinely
evolved separated-front controls. Unit tests also cover face/body diagonal
surfaces, signed entropy interpolation, a missing domain exit, and the search
limit. Negative profiles and broadened exact references are controlled inputs,
not new exact hydrodynamic evolutions.

The numerical results and pass/fail gates are recorded in
`outputs/transition_jump_screening_correction/verification.json`. Source and
input fingerprints are checked at the end of the replay. Historical source
hashes in the original experiment are retained as history; the intentionally
modified detector files are fingerprinted separately for this correction.

The replay recovers the 4/12/24 archived misses at 512/1024/2048 cells,
respectively, and loses none of the previously located supported fronts. All
284 above-supported observations are located in the new mode. Default masks
and floating-point fields match the archive exactly at the 142 replayed times.
These are conditional supported observations, not a completeness denominator
for every shock in every saved state. See the output README and CSVs for the
control counts and the remaining unresolved close-front cases.

All final replay gates pass: 112 basic controls (including 68 negative cases
with zero reported surfaces), 84 opposing/isolated-branch cases with no
regression among previously accurate cases, and all 276 required separated
front checks from the evolved collision archive. This does not mean all
sub-cell or broadened close-front cases pass; their failures are retained.

## What this does and does not establish

This targets the measured short-baseline screening failure. It leaves the
temperature direction and alignment test in place, so it cannot recover cells
that fail that earlier test. Independent motion/jump labels remain conditional
on sampled states and their declared sensitivity; they are not exact truth.
Close or unresolved structures can still defeat detection or reconstruction.

Before enabling this mode in multi-bubble production runs, compare it with the
saved bubble gas and retain the existing temperature-alignment diagnosis where
it still applies. Curved-flow completeness, callback/runtime overhead and the
old benchmark numbers have not been revalidated for this additional ray search.

## Reproduce

```sh
MPLCONFIGDIR=/private/tmp/master-thesis-mpl-benchmark \
python -m experiments.shock.check_transition_jump_screening \
  --output-dir outputs/transition_jump_screening_correction

python -m unittest discover -s tests/physics_modules/shock_finder
```

Use a fresh output directory and compatible project Python. This planar replay
does not require a remote GPU. The finite-blast and evolved-collision archives
from the preceding steps must be present.
